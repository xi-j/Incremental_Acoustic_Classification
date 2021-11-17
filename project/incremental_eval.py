import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix

from src.wav2clip_classifier import w2c_classifier
from src.CNNs import Cnn14, Cnn6
from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
from src.replay import Replay, ReplayExposureBlender, classwise_accuracy
from src.novelty_detect import make_novelty_detector

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="test_config")
    args = vars(parser.parse_args())
    
    from configs import *
    cfg = globals()[args['cfg']]()

    hyperparams = cfg['hyperparams']

    experiment_name = cfg['experiment_name']
    
    urban_eval = UrbanSoundDataset(cfg['dataset_path'], hyperparams['eval_folder'], sr=hyperparams['sr'], transform=ToTensor())
    urban_eval_loader = DataLoader(urban_eval, batch_size=8, shuffle=False,num_workers=4)

    device = hyperparams['device']
    
    MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"

    scenario = 'finetune'
    ckpt = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device, progress=True)
    model =w2c_classifier(ckpt=ckpt, scenario=scenario)
    model.to(device)

    model.load_state_dict(
        torch.load('ckpts/incremental_train_1637103126/exposure20/Wav2CLIP9_8_7_1_0_2_3_4_5_6_0.9690909090909091.pt')
        )

    truths = []
    predictions = []
        
    model.eval()
    with torch.no_grad():
        for x, labels in urban_eval_loader:
            x, labels = x.to(device), labels.to(device)

            predicts = model(x)
            #print(predicts.shape)
            predicts = torch.argmax(predicts,1)

            truths.extend(labels.tolist())
            predictions.extend(predicts.tolist())

    truths = np.array(truths)
    predictions = np.array(predictions)

    # 3 - 0,8 - 3, 9 - 6, 7 - 7, 0 - 8, 6 - 9
    
    mapping = {9:9, 8:8, 7:7, 1:1, 0:3, 2:2, 3:6, 4:4, 5:0, 6:5, 10:10}

    for i in range(len(predictions)):
        predictions[i] = mapping[predictions[i]]

    acc, classwise_acc = classwise_accuracy(np.array(predictions).flatten(),
                                                    np.array(truths).flatten(),
                                                    11,
                                                    [0,1,2,3,4,5,6,7,8,9]
                                                    )
    

    print(acc)
    print(classwise_acc)

    matrix = np.array(confusion_matrix(truths, predictions, labels=range(11)))

    print(matrix)
    print(np.sum(matrix, 1))

    # 3 - 0,8 - 3, 9 - 6, 7 - 7, 0 - 8, 6 - 9


    """
    0.7156511350059738
    [0.38       0.90909091 0.91       0.76       0.49       0.84946237
    0.9375     0.70833333 0.59036145 0.89      ]
    [[38  1  6  0  0 27  0  0  0 28  0]
    [ 0 30  0  0  0  0  0  0  1  2  0]
    [ 0  0 91  0  2  2  0  0  4  1  0]
    [ 0  1  6 76  9  0  1  0  2  5  0]
    [ 0 13  5  1 49  5  1  0  3 23  0]
    [ 4  0  1  0  3 79  0  5  0  1  0]
    [ 0  0  0  0  2  0 30  0  0  0  0]
    [ 0  8  0  0  0 11  0 68  1  8  0]
    [ 0  0 25  5  3  0  0  0 49  1  0]
    [ 0  0  8  0  3  0  0  0  0 89  0]
    [ 0  0  0  0  0  0  0  0  0  0  0]]
    """