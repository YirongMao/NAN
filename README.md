## NAN (Neural Aggregation Network for Video Face Recognition)

I have reproduced NAN on IJB-A, Celebrity-1000 datasets. In this repository I will show how to train and test on IJB-A dataset.

## Result
Average Pooling (AVE) face features within a template as the Baseline method.

AVE: TAR=[39.78, 64.20, 84.143135, 96.04231 ]@FAR=[1e-4, 1e-3, 1e-2, 1e-1]

NAN: TAR=[54.51, 74.54, 87.66, 96.31]@FAR=[1e-4, 1e-3, 1e-2, 1e-1]

## Train and Test
1. Before to train the model, please download face features from [[BaiduYun]](https://pan.baidu.com/s/1LD3NTWsf0NVC4uYnBFlS7Q) (code: 7zyg) and put the unzipped data into ./data/IJBA/resnet34. The features are extracted by ResNet34 model trained on WebFace dataset with only Softmax Loss.

2. run ./prepare_data/dataset_IJBA.py to prepare the train and test data for IJB-A dataset.

3. run ./train_ijba.py to train and test on IJB-A dataset

## Train and Test Log
You can find the train log at [[log]](https://github.com/YirongMao/NAN/blob/master/data/IJBA/model/model_resnet34_s8_NAN/train_s8b128_.txt) and [[eval_result]](https://github.com/YirongMao/NAN/blob/master/data/IJBA/model/model_resnet34_s8_NAN/eval_result.txt)

## Contact
If you find any bug, please be free to contact me. My email is yirong.maoATvipl.ict.ac.cn


