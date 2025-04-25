import random
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import sys
import time
from copy import deepcopy
from sentence_transformers import SentenceTransformer
import boto3
import io


# fore-deal
def remove_infrequent_items(df, min_counts=5):
  counts = df['asin'].value_counts()
  df = df[df["asin"].isin(counts[counts >= min_counts].index)]
  print("items with < {} interactoins are removed".format(min_counts))
  return df

def remove_infrequent_users(df, min_counts=10):
  counts = df['reviewerID'].value_counts()
  df = df[df["reviewerID"].isin(counts[counts >= min_counts].index)]
  print("users with < {} interactoins are removed".format(min_counts))
  return df

# select user session >= 4
def select_sessions(df, mins, maxs):
  users = df['reviewerID'].unique()
  counter = 0
  allcount = len(users)
  selected_id = []
  for reviewerID, group in df.groupby('reviewerID'):
    counter += 1
    time_len = len(group['unixReviewTime'].unique())
    if time_len >= mins and time_len <= maxs:
      selected_id.append(reviewerID)

    sys.stdout.write('Session select: {:.2f}%\r'.format(100 * counter / allcount))
    sys.stdout.flush()
    time.sleep(0.01)
  df = df[df['reviewerID'].isin(selected_id)]
  print('selected session({0} <= session <= {1}):{2}'.format(mins, maxs, len(df)))
  return df

# select from meta_df
def select_meta(df, meta_df):
  items = df['asin'].unique()
  return meta_df[meta_df['asin'].isin(items)]


def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key


random.seed(1234)

# Create a boto3 S3 client
s3 = boto3.client('s3')
dataset_list = ['CDs_and_Vinyl', 'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Office_Products', 'Movies_and_TV', 'Beauty', 'Home_and_Kitchen', 'Video_Games', 'Toys_and_Games', 'Books', 'Electronics']
bucket = 'your-s3-bucket'

for dataset in dataset_list:
    print(f"Processing dataset: {dataset}")
    # S3 configuration
    object_key = f'data/{dataset}_reviews.pkl'
    buffer = io.BytesIO()
    
    # Download the pickle file into memory
    s3.download_fileobj(bucket, object_key, buffer)
    buffer.seek(0)
    
    # Load the pickle object
    reviews_df = pickle.load(buffer)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime', 'reviewText', 'overall', 'summary']]
    reviews_df['unixReviewTime'] = reviews_df['unixReviewTime'] // 3600 // 24
    
    # S3 configuration
    object_key = f'data/{dataset}_meta.pkl'
    buffer = io.BytesIO()
    
    # Download the pickle file into memory
    s3.download_fileobj(bucket, object_key, buffer)
    buffer.seek(0)
    # Load the pickle object
    meta_df = pickle.load(buffer)
    meta_df = meta_df[['asin', 'categories', 'description', 'title', 'price', 'brand']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
    
    reviews_df = remove_infrequent_users(reviews_df, 10)
    reviews_df = remove_infrequent_items(reviews_df, 8)
    reviews_df = select_sessions(reviews_df, 4, 90)
    meta_df = select_meta(reviews_df, meta_df)
    columns = ['asin', 'categories', 'description', 'title']
    reviews_df = reviews_df.merge(meta_df[columns], on='asin', how='left')
    print('num of users:{}, num of items:{}'.format(len(reviews_df['reviewerID'].unique()), len(reviews_df['asin'].unique())))
    print('Select all done.')
    
    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')
    
    user_count, item_count, cate_count, example_count =\
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))
    
    
    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    
    item_cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
    item_cate_list = np.array(item_cate_list, dtype=np.int32)
    
    
    # Load the MiniLM model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Ensure there are no NaNs in the description column
    meta_df['description'] = meta_df['description'].fillna('')
    
    # Encode each description into a dense vector
    description_embedding = model.encode(meta_df['description'].tolist(), convert_to_numpy=True, device='cpu', batch_size=64)

    # Create a BytesIO buffer
    buffer = io.BytesIO()
    
    # Dump pickle content to the buffer
    pickle.dump((reviews_df, meta_df), buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(item_cate_list, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump(description_embedding, buffer, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count), buffer, pickle.HIGHEST_PROTOCOL)
    
    # Reset buffer's position to the beginning
    buffer.seek(0)
    
    # Upload to S3
    s3_key = f'processed_data/{dataset}_remap.pkl'
    print(f"Upload processed data to s3...")
    s3.upload_fileobj(buffer, bucket, s3_key)
    print("==============================")
    
