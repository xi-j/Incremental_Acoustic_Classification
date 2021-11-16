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
        torch.load('ckpts/incremental_train_1637024110/exposure11/Wav2CLIP6_5_4_9_0_1_2_3_7_8_0.948.pt')
        )


    truths = []
    predictions = []
        
    model.eval()
    with torch.no_grad():
        for x, labels in urban_eval_loader:
            x, labels = x.to(device), labels.to(device)

            predicts = model(x)
            predicts = torch.argmax(predicts,1)

            truths.extend(labels.tolist())
            predictions.extend(predicts.tolist())

    truths = np.array(truths)
    predictions = np.array(predictions)

    # 3 - 0,8 - 3, 9 - 6, 7 - 7, 0 - 8, 6 - 9
    mapping = {0:3, 3:8, 6:9, 7:7, 8:0, 9:6}
    for i in range(len(predictions)):
        if predictions[i] in [2,4,1,5]:
            continue
        else:
            predictions[i] = mapping[predictions[i]]

    acc = sum(truths==predictions)/len(truths)

    print('Overrall Accuracy: ', acc)

    for l in range(10):
        label_truths = truths[truths == l]
        label_predictions = predictions[truths == l]
        label_acc = sum(label_truths==label_predictions)/len(label_truths)
        print('Class {} Accuracy:'.format(l), label_acc)

    matrix = np.array(confusion_matrix(truths, predictions, labels=range(10)))

    print(matrix)
    print(np.sum(matrix, 1))

    # 3 - 0,8 - 3, 9 - 6, 7 - 7, 0 - 8, 6 - 9