# ACMINet

The Pytorch implementation of *"A 3D Cross-Modality Feature Interaction Network With Volumetric Feature Alignment for Brain Tumor and Tissue Segmentation"* on the BrainTS2020 dataset.




## Requirements

Experiments were performed on an Ubuntu 18.04 workstation with two 24G NVIDIA GeForce RTX 3090 GPUs , CUDA 11.1, and install the virtual environment (python3.8) by:

```
pip install -r requirements.txt
```

The Ranger optimizer is used in this project, and install it by the following project:

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer



## Implementation

### Data download

Download the [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) dataset and change data config:

```
vim ./src/path_config.py
# training dataset
BRATS_TRAIN_FOLDERS = "./xx/MICCAI_BraTS2020_TrainingData/"
# validation dataset
BRATS_VAL_FOLDER = "./xx/MICCAI_BraTS2020_ValidationData/"
# test dataset 
BRATS_TEST_FOLDER = "./xx/MICCAI_BraTS2020_Data_Testing/"

```



### Training and Predicting

Run different models and change the '--fold' parameters to select the cross-validation fold (0 to 4)：

```
sh train_models.sh
```

Predicting the validation data by multi-model ensemble:

```
python step2_inference.py --devices 0 --on val --tta
```



## Citation

If our projects are beneficial for your works, please cite:

```
@ARTICLE{9920184,
author={Zhuang, Yuzhou and Liu, Hong and Song, Enmin and Hung, Chih-Cheng},
journal={IEEE Journal of Biomedical and Health Informatics}, 
title={A 3D Cross-Modality Feature Interaction Network With Volumetric Feature Alignment for Brain Tumor and Tissue Segmentation}, 
year={2023},
volume={27},
number={1},
pages={75-86},
doi={10.1109/JBHI.2022.3214999}}
```



## Acknowledge

Our training codes is developed from the [Top10 BraTS 2020 open sourced solution](https://github.com/lescientifik/open_brats2020), please refer to their paper: https://arxiv.org/abs/2011.01045.

