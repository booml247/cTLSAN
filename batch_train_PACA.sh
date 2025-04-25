#!/bin/bash

cd PACA || exit

dataset_list=(
  "CDs_and_Vinyl"
  "Clothing_Shoes_and_Jewelry"
  "Digital_Music"
  "Office_Products"
  "Movies_and_TV"
  "Beauty"
  "Home_and_Kitchen"
  "Video_Games"
  "Toys_and_Games"
  "Books"
  "Electronics"
)

python build_dataset.py

for dataset in "${dataset_list[@]}"
do
  echo "🚀 Training on dataset: $dataset"
  python train.py --model_dir "save_path_$dataset" --dataset "$dataset"
done
