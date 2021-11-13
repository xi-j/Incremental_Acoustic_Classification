import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.CNNs import Cnn14, Cnn6
from src.classifier import SimpleClassifier, Wav2CLIPClassifier
from src.wav2clip_classifier import w2c_classifier
from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
from src.replay import Replay, ReplayExposureBlender, classwise_accuracy

hyperparams = {
    'sr': 16000,
    'exposure_size': 300,
    'exposure_val_size': 50, 
    'initial_K': 4,
    'batch_size': 4,
    'num_epochs': 30,
    'num_epochs_ex': 10,
    'replay_tr': 50,
    'replay_val': 20,
    'lr': 5e-6,
    'model': 'Wav2CLIP'
}
hyperparams['exposure_tr_size'] = hyperparams['exposure_size'] - hyperparams['exposure_val_size']

exposure_generator = UrbanSoundExposureGenerator(
    'UrbanSound8K', 
    range(1, 10), 
    sr=hyperparams['sr'], 
    exposure_size=hyperparams['exposure_size'], 
    exposure_val_size=hyperparams['exposure_val_size'], 
    initial_K=hyperparams['initial_K']
)

# Intitial Training set
initial_tr, initial_val, seen_classes = exposure_generator.get_initial_set()

print('Initial seen classes: ', seen_classes)
print('Number of samples in initial_tr: ', len(initial_tr))
print('Number of samples in initial_val: ', len(initial_val))

# Initial Replay
replay_tr = Replay(initial_tr, seen_classes, hyperparams['replay_tr'])
replay_val = Replay(initial_val, seen_classes, hyperparams['replay_val'])

# Exposure List
exposure_tr_list = []
exposure_val_list = []
exposure_label_list = []
for i in range(len(exposure_generator)):
    exposure_tr, exposure_val, label  = exposure_generator[i]  
    exposure_tr_list.append(exposure_tr)
    exposure_val_list.append(exposure_val)
    exposure_label_list.append(label)
    
print(f'Initial Replay: ')
print('Number of samples in replay_tr: ', len(replay_tr))
print('Seen classes in replay_tr: ', replay_tr.classes)
print('Number of samples in replay_val: ', len(replay_val))
print('Seen classes in replay_tr: ', replay_tr.classes)

# Feed exposure and update replay
for i, label in enumerate(exposure_label_list):
    print(f'Exposure No.{i}')
    exposure_tr = exposure_tr_list[i]
    exposure_val = exposure_val_list[i]  
    
    if label not in seen_classes:
        for i in range(10):
            if i not in seen_classes:
                pseudo_label = i
                break
        seen_classes.append(pseudo_label)

        replay_tr.update(exposure_tr, pseudo_label)
        replay_val.update(exposure_val, pseudo_label)

    else:
        inferred_label = label
        
        replay_tr.update(exposure_tr, inferred_label)
        replay_val.update(exposure_val, inferred_label)


    print('Number of samples in replay_tr: ', len(replay_tr))
    print('Seen classes in replay_tr: ', replay_tr.classes)
    print('Number of samples in replay_val: ', len(replay_val))
    print('Seen classes in replay_val: ', replay_tr.classes)



# exposure_tr = exposure_tr_list[0]
# label = exposure_label_list[0]
# new_tr = ReplayExposureBlender(replay_tr, exposure_tr, seen_classes, 
#                                resize=hyperparams['exposure_tr_size'])
# print('Number of samples in new_tr: ', len(new_tr))