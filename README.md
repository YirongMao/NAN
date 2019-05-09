# NAN (Neural Aggregation Network for Video Face Recognition)

I have reproduced NAN on IJB-A, Celebrity-1000 datasets. I show how to train and test on IJB-A dataset in this repository 

# Result
Average Pooling (AVE) faces features within a template as the Baseline method.

AVE:

NAN:

# To Train
1. Before to train the model, please download the face features from BaiduYun() and put the unzipped data in the directory of ./data/IJBA/. The features are extracted by ResNet34 model trained on WebFace dataset with only Softmax Loss.

2. run ./prepare_data/dataset_IJBA.py to prepare the train and test data for IJB-A dataset.

3. run ./train_ijba.py to train and test on IJB-A dataset

# Train Log
You can find the train log at 

## If you find any bug, please be free to contact me. My email is yirong.maoATvipl.ict.ac.cn


