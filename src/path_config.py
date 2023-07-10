import os
from pathlib import Path


BRATS_TRAIN_FOLDERS = "/mnt/data1/zyzbrain/Datasets/BrainTumor/brats2020/MICCAI_BraTS2020_TrainingData/"
BRATS_VAL_FOLDER = "/mnt/data1/zyzbrain/Datasets/BrainTumor/brats2020/MICCAI_BraTS2020_ValidationData/"
BRATS_TEST_FOLDER = "/mnt/data1/zyzbrain/Datasets/BrainTumor/brats2020/MICCAI_BraTS2020_ValidationData/"




def get_brats_folder(on="val"):
    if on == "train":
        return os.environ['BRATS_FOLDERS'] if 'BRATS_FOLDERS' in os.environ else BRATS_TRAIN_FOLDERS
    elif on == "val":
        return os.environ['BRATS_VAL_FOLDER'] if 'BRATS_VAL_FOLDER' in os.environ else BRATS_VAL_FOLDER
    elif on == "test":
        return os.environ['BRATS_TEST_FOLDER'] if 'BRATS_TEST_FOLDER' in os.environ else BRATS_TEST_FOLDER
