import random
import pickle
import numpy as np
import pandas as pd
import copy
import boto3 
import io

max_length = 90
random.seed(1234)
# Create a boto3 S3 client
s3 = boto3.client('s3')
# dataset_list = ['CDs_and_Vinyl', 'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Office_Products', 'Movies_and_TV', 'Beauty', 'Home_and_Kitchen', 'Video_Games', 'Toys_and_Games', 'Books', 'Electronics']
dataset_list = ['Digital_Music']

bucket = 'search-page-template-datasets-devo'

for dataset in dataset_list:
    object_key = f'playground/liangsiq/LLM/processed_data/{dataset}_remap.pkl'
    buffer = io.BytesIO()
    
    # Download the pickle file into memory
    s3.download_fileobj(bucket, object_key, buffer)
    buffer.seek(0)
    # Load the pickle object
    reviews_df, meta_df = pickle.load(buffer)
    item_cate_list = pickle.load(buffer)
    description_embedding = pickle.load(buffer)
    user_count, item_count, cate_count, example_count = pickle.load(buffer)
        
    # generate current time positions
    gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    
    def proc_time_emb(hist_t, cur_t):
        # older timestamps get lower weights
        hist_t = [cur_t - i + 1 for i in hist_t]
        hist_t = [1/np.sum(i >= gap) for i in hist_t]
        return hist_t
    
    # # Safe fallback
    # def normalize_text_input(text):
    #     if isinstance(text, list):
    #         # Replace None or non-str with empty string
    #         return [t if isinstance(t, str) else "" for t in text]
    #     elif isinstance(text, str):
    #         return text
    #     else:
    #         return ""  # For unexpected types like float('nan'), None, etc.
    
    train_set = []
    test_set = []
    for reviewerID, hist in reviews_df.groupby('reviewerID'):
        pos_list = hist['asin'].tolist()
        tim_list = hist['unixReviewTime'].tolist()
        review_list = hist['description'].to_list()
        rate_list = hist['overall'].to_list()
        summary_list = hist['summary'].to_list()    
        # Clean the list in case it contains None
        cleaned_review_list = [r if isinstance(r, str) else "" for r in review_list]
        
        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg
        neg_list = [gen_neg() for i in range(len(pos_list))]
        # Convert meta_df to a dictionary for fast lookup
        asin_to_description = dict(zip(meta_df['asin'], meta_df['description']))
        
        # Finally, get the description for each asin
        neg_reviews = [asin_to_description.get(asin, "") for asin in neg_list]

        length = len(pos_list)
        valid_length = min(length, max_length)
        i = 0
        tim_list_session = list(set(tim_list))
        tim_list_session.sort()
        pre_session = []
        pre_time = []
        pre_cates = []
        pre_review = []
        for t in tim_list_session:
            count = tim_list.count(t)
            new_session = pos_list[i:i+count]
            new_time = tim_list[i:i+count]
            new_cates = [meta_df[meta_df['asin'] == item]['categories'].tolist()[0] for item in new_session]
            new_review = cleaned_review_list[i:i+count]
            new_rate = rate_list[i:i+count]
            new_summary = summary_list[i:i+count]
    
            if t == tim_list_session[0]:
                pre_session.extend(new_session)
                pre_time.extend(new_time)
                pre_cates.extend(new_cates)
                pre_review.extend(new_review)
            else:
                now_cate = pd.value_counts(pre_cates).index[0]
                if i+count < valid_length-1:
                    pre_time_emb = proc_time_emb(pre_time, tim_list[i])
                    pre_session_copy = copy.deepcopy(pre_session)
                    pre_review_copy = copy.deepcopy(pre_review)
                    train_set.append((reviewerID, pre_review_copy, pre_session_copy, new_review, new_session, pre_time_emb, cleaned_review_list[i+count], pos_list[i+count], 1, now_cate))
                    train_set.append((reviewerID, pre_review_copy, pre_session_copy, new_review, new_session, pre_time_emb, neg_reviews[i+count], neg_list[i+count], 0, now_cate))
                    pre_session.extend(new_session)
                    pre_time.extend(new_time)
                    pre_cates.extend(new_cates)
                    pre_review.extend(new_review)
                else:
                    pos_item = pos_list[i]
                    if count > 1:
                        pos_item = random.choice(new_session)
                        new_session.remove(pos_item)
                    neg_index = pos_list.index(pos_item)
                    pos_neg = (pos_item, neg_list[neg_index])
                    pos_neg_review = (cleaned_review_list[neg_index], neg_reviews[neg_index])
                    pre_time_emb = proc_time_emb(pre_time, t)
                    test_set.append((reviewerID, pre_review, pre_session, new_review, new_session, pre_time_emb, pos_neg_review, pos_neg, now_cate))
                    break
            i += count
    
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    assert len(test_set) == user_count

    # Create a BytesIO buffer
    buffer = io.BytesIO()
    pickle.dump(train_set, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(item_cate_list, buffer, pickle.HIGHEST_PROTOCOL)

    # Reset buffer's position to the beginning
    buffer.seek(0)
    
    # Upload to S3
    s3_key = f'playground/liangsiq/LLM/LLM_input_data/{dataset}_dataset.pkl'
    print(f"Upload postprocessed data to s3...")
    s3.upload_fileobj(buffer, bucket, s3_key)
    print("==============================")
    