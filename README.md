## NAN (Neural Aggregation Network for Video Face Recognition)

I have reproduced NAN on IJB-A, Celebrity-1000 datasets. In this repository I will show how to train and test on IJB-A dataset.

## Result
Average Pooling (AVE) faces features within a template as the Baseline method.

AVE:

NAN:

## Train and Test
1. Before to train the model, please download face features from BaiduYun(链接：https://pan.baidu.com/s/1LD3NTWsf0NVC4uYnBFlS7Q 
提取码：7zyg) and put the unzipped data into ./data/IJBA/. The features are extracted by ResNet34 model trained on WebFace dataset with only Softmax Loss.

2. run ./prepare_data/dataset_IJBA.py to prepare the train and test data for IJB-A dataset.

3. run ./train_ijba.py to train and test on IJB-A dataset

## Train and Test Log
You can find the train log at 

## Contact
If you find any bug, please be free to contact me. My email is yirong.maoATvipl.ict.ac.cn


