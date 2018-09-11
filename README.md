# A PyTorch implementation of V-Net

This V-Net code is a [PyTorch](http://pytorch.org/) implementation of the paper
[V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
by Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. The data preprocessing code is based on the official version [faustomilletari/VNet](https://github.com/faustomilletari/VNet), while the V-Net model was built on [mattmacy/vnet.pytorch](https://github.com/mattmacy/vnet.pytorch).

To apply the code, just modify the lines from "main.py", "DataManager.py" and "train.py" where marked as "require customization"

Average dice coefficient for the 30 test cases: 0.887.
PROMISE12 challenge score: 85.67
Rank: #23 on Sep 11, 2018.
