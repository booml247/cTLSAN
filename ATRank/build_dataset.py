import random
import pickle
import numpy as np
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
    
    gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    
    def proc_time_emb(hist_t, cur_t):
      hist_t = [cur_t - i + 1 for i in hist_t]
      hist_t = [np.sum(i >= gap) for i in hist_t]
      return hist_t
    
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
    
      valid_length = min(len(pos_list), max_length)
      for i in range(1, valid_length):
        hist_i = pos_list[:i]
        hist_t = proc_time_emb(tim_list[:i], tim_list[i])
        if i != valid_length - 1:
          train_set.append((reviewerID, hist_i, hist_t, pos_list[i], 1))
          train_set.append((reviewerID, hist_i, hist_t, neg_list[i], 0))
        else:
          label = (pos_list[i], neg_list[i])
          test_set.append((reviewerID, hist_i, hist_t, label))
    
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    assert len(test_set) == user_count
     
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    pickle.dump(train_set, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), buffer, pickle.HIGHEST_PROTOCOL)
    
    # Reset buffer's position to the beginning
    buffer.seek(0)
    
    # Upload to S3
    s3_key = f'ATRank_input_data/{dataset}_dataset.pkl'
    print(f"Upload postprocessed data to s3...")
    s3.upload_fileobj(buffer, bucket, s3_key)
    print("==============================")