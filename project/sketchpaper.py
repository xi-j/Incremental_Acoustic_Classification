from comet_ml import Experiment
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse

from src.wav2clip_classifier import w2c_classifier
from src.CNNs import Cnn14, Cnn6
from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
from src.replay import ReplayExposureBlender, classwise_accuracy
from src.novelty_detect import make_novelty_detector

if __name__ == '__main__':
    
    # init hyperparameters, comet, ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="test_config")
    args = vars(parser.parse_args())
    
    from configs import *
    cfg = globals()[args['cfg']]()

    hyperparams = cfg['hyperparams']


    #####################################################################################


    # init initial train/val set, exposures
    exposure_generator = UrbanSoundExposureGenerator(
        cfg['dataset_path'], 
        hyperparams['train_val_folders'], 
        sr=hyperparams['sr'], 
        exposure_size=hyperparams['exposure_size'], 
        exposure_val_size=hyperparams['exposure_val_size'], 
        initial_K=hyperparams['initial_K']
    )

    prev_tr, prev_val, seen_classes = exposure_generator.get_initial_set()

    exposure_tr_list = []
    exposure_val_list = []
    exposure_label_list = []

    for i in range(len(exposure_generator)):
        exposure_tr, exposure_val, label  = exposure_generator[i]  
        exposure_tr_list.append(exposure_tr)
        exposure_val_list.append(exposure_val)
        exposure_label_list.append(label)

    
    new_set = ReplayExposureBlender(prev_tr, exposure_tr_list[2], seen_classes, resize=125)

    count = {}

    for x, label in new_set:
        if label not in count:
            count[label] = 0
        else:
            count[label] += 1

    print(count)