# C-TLSAN
This is the implementation for our paper: C-TLSAN: Contextual Time-aware Long- and Short-term
Attention Network for Personalized Recommendation. Here are the brief introductions to the dataset and the experiment results. 

## Environments
boto3==1.36.23

numpy==2.2.5

pandas==2.2.3

sentence_transformers==4.1.0

tensorflow==2.17.0

## Datasets
Amazon exposes the official datasets (http://jmcauley.ucsd.edu/data/amazon/) which have filtered out users and items with less than 5 reviews and removed a large amount of invalid data. Because of above advantages, these datasets are widely utilized by researchers. We also chose Amazon's dataset for experiments. In our experiments, only users, items, interactions, and category information are utilized. We do the preprocessing in the following two steps:
1. Remove the users whose interactions less than 10 and the items which interactions less than 8 to ensure the effectiveness of each user and item.
2. Select the users with more than 4 sessions, and select up to 90 behavior records for the remaining users. This step guarantees the existence of long- and short-term behavior records and all behavior records occurred within recent three months.

### Statistics (after preprocessing)
Datasets | users | items | categories | samples | avg.<br>items/cate | avg.<br>behaviors/item | avg.<br>behaviors/user
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:
Electronics | 39991 | 22048 | 673 | 561100 | 32.8 | 25.4 | 14.0
CDs-Vinyl | 24179 | 27602 | 310 | 470087 | 89.0 | 17.0 | 19.4
Clothing-Shoes | 2010 | 1723 | 226 | 13157 | 7.6 | 7.6 | 6.5
Digital-Music | 1659 | 1583 | 53 | 28852 | 29.9 | 18.2 | 17.4
Office-Products | 1720 | 901 | 170 | 29387 | 5.3 | 32.6 | 17.0
Movies-TV | 35896 | 28589 | 15 | 752676 | 1905.9 | 20.9 | 26.3
Beauty | 3783 | 2658 | 179 | 54225 | 14.8 | 20.4 | 14.3
Home-Kitchen | 11567 | 7722 | 683 | 143088 | 11.3 | 12.3 | 18.5
Video-Games | 5436 | 4295 | 58 | 83748 | 74.1 | 19.5 | 15.4
Toys-and-Games | 2677 | 2474 | 221 | 37515 | 11.2 | 15.2 | 14.0

## Experiment results
|                            | LLM   | CSAN  | ATRank | Bi-LSTM | PACA  | TLSAN | cTLSAN |
|----------------------------|-------|-------|--------|---------|-------|-------|--------|
| CDs_and_Vinyl              | 0.827 | 0.813 |  0.889 |   0.881 | 0.801 | 0.942 |  0.938 |
| Clothing_Shoes_and_Jewelry | 0.639 | 0.761 |  0.663 |   0.668 | 0.798 | 0.927 |  0.938 |
| Digital-Music              | 0.808 | 0.729 |  0.825 |   0.792 | 0.963 | 0.972 |  0.974 |
| Office_Products            | 0.516 | 0.824 |  0.921 |   0.856 | 0.910 | 0.969 |  0.976 |
| Movies_and_TV_5            | 0.813 | 0.797 |  0.860 |   0.824 | 0.806 | 0.879 |  0.909 |
| Beauty                     | 0.618 | 0.727 |  0.806 |   0.773 | 0.859 | 0.925 |  0.947 |
| Home_and_Kitchen           | 0.592 | 0.702 |  0.736 |   0.684 | 0.788 | 0.865 |  0.895 |
| Video_Games                | 0.587 | 0.807 |  0.870 |   0.820 | 0.917 | 0.914 |  0.933 |
| Toys_and_Games             | 0.678 | 0.812 |  0.829 |   0.775 | 0.861 | 0.922 |  0.936 |
| Electronics                | 0.587 | 0.811 |  0.841 |   0.811 | 0.835 | 0.894 |  0.913 |

## How to run the codes
Download raw data and preprocess it with utils:
```
cd utils
sh 0_download_raw.sh
python3 1_convert_pd.py
python3 2_remap_id.py
```
Train and evaluate the model, take cTLSAN as an example:
```
bash bath_train_cTLSAN.sh
```

