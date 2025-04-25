import pickle
import pandas as pd
import boto3
import io

def to_df(file_path):
      with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
          df[i] = eval(line)
          i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def s3_to_df(bucket, key):
    # Download file from S3
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    
    # Read JSON lines from buffer
    df = {}
    i = 0
    for line in buffer:
        decoded_line = line.decode('utf-8').strip()
        if decoded_line:  # skip empty lines
            df[i] = eval(decoded_line)
            i += 1
    
    df = pd.DataFrame.from_dict(df, orient='index')
    return df
          
s3 = boto3.client('s3')
bucket = 'your-s3-bucket'
dataset_list = ['CDs_and_Vinyl', 'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Office_Products', 'Movies_and_TV', 'Beauty', 'Home_and_Kitchen', 'Video_Games', 'Toys_and_Games', 'Books', 'Electronics']
for dataset in dataset_list:
    print(f"Processing dataset: {dataset}")
    review_data = f'reviews_{dataset}_5.json'
    meta_data = f'meta_{dataset}.json'

    load_path = 'raw_data/'
    
    reviews_df = s3_to_df(bucket, load_path+review_data)
    # Upload to S3
    save_key = f'data/{dataset}_reviews.pkl' 
    
    # Convert to bytes using pickle
    pickle_buffer = io.BytesIO()
    pickle.dump(reviews_df, pickle_buffer)
    pickle_buffer.seek(0)

    print(f"Upload review data to s3...")
    s3.upload_fileobj(pickle_buffer, bucket, save_key)
    
    # with open('../raw_data/reviews.pkl', 'wb') as f:
    #   pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    
    meta_df = s3_to_df(bucket, load_path+meta_data)
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    save_key = f'data/{dataset}_meta.pkl'  # Optional folder
    # Convert to bytes using pickle
    pickle_buffer = io.BytesIO()
    pickle.dump(meta_df, pickle_buffer)
    pickle_buffer.seek(0)

    print(f"Upload meta data to s3...")
    s3.upload_fileobj(pickle_buffer, bucket, save_key)
    print('================================================')
    
    # with open('../raw_data/meta.pkl', 'wb') as f:
    #   pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
