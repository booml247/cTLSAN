import random
import pickle
import numpy as np
import copy
import boto3
import io

max_length = 90
random.seed(1234)

# Create a boto3 S3 client
s3 = boto3.client('s3')
dataset_list = ['CDs_and_Vinyl', 'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Office_Products', 'Movies_and_TV', 'Beauty', 'Home_and_Kitchen', 'Video_Games', 'Toys_and_Games', 'Books', 'Electronics']
bucket = 'your-s3-bucket'

for dataset in dataset_list:
    object_key = f'processed_data/{dataset}_remap.pkl'
    buffer = io.BytesIO()
    
    # Download the pickle file into memory
    s3.download_fileobj(bucket, object_key, buffer)
    buffer.seek(0)

    reviews_df, meta_df = pickle.load(buffer)
    cate_list = pickle.load(buffer)
    description_embedding = pickle.load(buffer)
    user_count, item_count, cate_count, example_count = pickle.load(buffer)
    
    train_set = []
    test_set = []
    for reviewerID, hist in reviews_df.groupby('reviewerID'):
      pos_list = hist['asin'].tolist()
      tim_list = hist['unixReviewTime'].tolist()
    
      def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
          neg = random.randint(0, item_count-1)
        return neg
      neg_list = [gen_neg() for i in range(len(pos_list))]
    
      length = len(pos_list)
      valid_length = min(length, max_length)
      i = 0
      tim_list_session = list(set(tim_list))
      tim_list_session.sort()
      pre_session = []
      pre_time = []
      for t in tim_list_session:
        count = tim_list.count(t)
        new_session = pos_list[i:i+count]
    
        if t == tim_list_session[0]:
          pre_session.extend(new_session)
        else:
          if i+count < valid_length-1:
            pre_session_copy = copy.deepcopy(pre_session)
            train_set.append((pre_session_copy, pos_list[i+count], 1))
            train_set.append((pre_session_copy, neg_list[i+count], 0))
            pre_session.extend(new_session)
          else:
            pos_item = pos_list[i]
            if count > 1:
              pos_item = random.choice(new_session)
              new_session.remove(pos_item)
            neg_index = pos_list.index(pos_item)
            pos_neg = (pos_item, neg_list[neg_index])
            test_set.append((pre_session, pos_neg))
            break
        i += count
    
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    assert len(test_set) == user_count

    # Create a BytesIO buffer
    buffer = io.BytesIO()
    pickle.dump(train_set, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count), buffer, pickle.HIGHEST_PROTOCOL)
    
    # Reset buffer's position to the beginning
    buffer.seek(0)
    
    # Upload to S3
    s3_key = f'PACA_input_data/{dataset}_dataset.pkl'
    print(f"Upload postprocessed data to s3...")
    s3.upload_fileobj(buffer, bucket, s3_key)
    print("==============================")
    