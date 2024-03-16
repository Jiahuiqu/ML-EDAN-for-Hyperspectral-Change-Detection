# ML-EDAN for Hyperspectral Change Detection
The implementation of the paper "A Multilevel Encoder-Decoder Attention Network for Change Detection in Hyperspectral Images" (IEEE Transactions on Geoscience and Remote Sensing 2022)

# Requirements
Ubuntu 20.04 cuda 11.0

Python 3.8 Pytorch 1.7

# Usage
We have presented test cases of the proposed model in model.py file.

# Hyperparameters
The training rate is set to 0.0001.

The batchsize is set to 16.

The patch size is set to 5.

The optimizer is Adam.

The more detailed training settings are shown in experiments of this paper.

# Reference
If you use this code, please kindly cite this
@ARTICLE{9624977,  
  &emsp;author={Qu, Jiahui and Hou, Shaoxiong and Dong, Wenqian and Li, Yunsong and Xie, Weiying},  
  &emsp;journal={IEEE Transactions on Geoscience and Remote Sensing},  
  &emsp;title={A Multilevel Encoder–Decoder Attention Network for Change Detection in Hyperspectral Images},  
  &emsp;year={2022},  
  &emsp;volume={60},  
  &emsp;number={},  
  &emsp;pages={1-13},  
  &emsp;doi={10.1109/TGRS.2021.3130122}}
