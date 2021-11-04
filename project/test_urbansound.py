from comet_ml import Experiment
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
from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
from src.replay import ReplayExposureBlender

experiment = Experiment(
    api_key="kOAHVqhBnkw2R6FQr6b0uOemJ",
    project_name="cs545",
    workspace="xi-j",
)
hyperparams = {
    'sr': 16000,
    'exposure_size': 300, 
    'exposure_val_size': 50, 
    'initial_K': 4,
    'batch_size': 4,
    'num_epochs': 30,
    'num_epochs_ex': 10,
    'lr': 5e-6,
    'model': 'Wav2CLIP'
}

experiment.log_parameters(hyperparams)
tags = 'UrbanSound', 'Wav2CLIP', 'Initial 4', 'Exposure 1+1', 'seen vs unseen'
experiment.add_tags(tags)
experiment.set_name(' '.join(tags))
############################################################################
def normalize_tensor_wav(x, eps=1e-10, std=None):
    mean = x.mean(-1, keepdim=True)
    if std is None:
        std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)

exposure_generator = UrbanSoundExposureGenerator(
    'UrbanSound8K', 
    range(1, 10), 
    sr=hyperparams['sr'], 
    exposure_size=hyperparams['exposure_size'], 
    exposure_val_size=hyperparams['exposure_val_size'], 
    initial_K=hyperparams['initial_K']
)

initial_tr, initial_val, seen_classes = exposure_generator.get_initial_set()
experiment.log_parameters({'inital_classes': seen_classes})

exposure_tr_list = []
exposure_val_list = []
exposure_label_list = []
for i in range(len(exposure_generator)):
    exposure_tr, exposure_val, label  = exposure_generator[i]  
    exposure_tr_list.append(exposure_tr)
    exposure_val_list.append(exposure_val)
    exposure_label_list.append(label)

initial_tr_loader = DataLoader(initial_tr, batch_size=4, shuffle=True, num_workers=4)
initial_val_loader = DataLoader(initial_val, batch_size=4, shuffle=True, num_workers=4)

############################################################################
device = torch.device('cuda:2')

if hyperparams['model'] == 'CNN14':
    model = Cnn14(sample_rate=16000, window_size=1024, hop_size=512, 
                  mel_bins=64, fmin=0, fmax=None, classes_num=10).to(device)

elif hyperparams['model'] == 'Wav2CLIP':
    model = Wav2CLIPClassifier()
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                             lr=hyperparams['lr'])

train_loss_list = []
val_acc_list = []
loss_counter = 0
loss_cycle = 50
best_acc = -1
acc = -1
# Stage 1: Pre-train the model with inital classes
with experiment.train():
    for epoch in tqdm(range(hyperparams['num_epochs']), desc='Epoch'):
        model.train()
        
        for x, y in tqdm(initial_tr_loader, desc='Training'):
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            train_loss_list.append(loss.item())
            loss_counter += 1
            if loss_counter % loss_cycle == 0:
                print()
                print('Average running loss:', sum(train_loss_list[-loss_cycle:]) / loss_cycle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictions = []
        truths = []
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(initial_val_loader, desc='Validating'):
                x, y = x.to(device), y.to(device)
                yhat = torch.argmax(model(x), 1)
                truths.extend(y.tolist())
                predictions.extend(yhat.tolist())
  
        acc = sum(np.array(truths)==np.array(predictions))/len(truths)
        print()
        print('Accuracy: ', acc)
        val_acc_list.append(acc)
        experiment.log_metric("Validation accuracy", acc, step=epoch)
        
last_acc = acc
torch.save(model.state_dict(), 
    (os.path.join('saved_models', 
    hyperparams['model']
    +'_'.join(map(str, seen_classes))
    +'_'+str(last_acc))+'.pt')
)

# Stage 2: Feed an exposure 
# Find a class we have already seen, and assign a new label to it
# Check how is the accuracy
for i, label in enumerate(exposure_label_list):
    if label in seen_classes:
        exposure_tr = exposure_tr_list[i]
        exposure_val = exposure_val_list[i]
        break
        
new_tr = ReplayExposureBlender(initial_tr, exposure_tr, seen_classes, label)
new_tr_loader = DataLoader(new_tr, batch_size=4, shuffle=True, num_workers=4)
           
model.load_state_dict(
    torch.load(os.path.join('saved_models', 
            hyperparams['model']
            +'_'.join(map(str, seen_classes))
            +'_'+str(last_acc)+'.pt')
    )
)
optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                             lr=hyperparams['lr'])

with experiment.train():   
    for epoch in tqdm(range(hyperparams['num_epochs_ex']), desc='Epoch'):
        model.train()
        for x, y in tqdm(new_tr_loader, desc='Training'):
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            train_loss_list.append(loss.item())
            loss_counter += 1
            if loss_counter % loss_cycle == 0:
                print()
                print('Average running loss:', sum(train_loss_list[-loss_cycle:]) / loss_cycle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        predictions = []
        truths = []
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(initial_val_loader, desc='Validating'):
                x, y = x.to(device), y.to(device)
                yhat = torch.argmax(model(x), 1)
                truths.extend(y.tolist())
                predictions.extend(yhat.tolist())
  
        acc = sum(np.array(truths)==np.array(predictions))/len(truths)
        print()
        print('Accuracy: ', acc)
        val_acc_list.append(acc)
        experiment.log_metric("Validation accuracy", acc, step=hyperparams['num_epochs']+epoch)            
        
# Find a class we have never seen, and assign a new label to it
# Check how is the accuracy
for i, label in enumerate(exposure_label_list):
    if label not in seen_classes:
        exposure_tr = exposure_tr_list[i]
        exposure_val = exposure_val_list[i]
        break
        
new_tr = ReplayExposureBlender(initial_tr, exposure_tr, seen_classes, label)
new_tr_loader = DataLoader(new_tr, batch_size=4, shuffle=True, num_workers=4)
           
model.load_state_dict(
    torch.load(os.path.join('saved_models', 
            hyperparams['model']
            +'_'.join(map(str, seen_classes))
            +'_'+str(last_acc)+'.pt')
    )
)
optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                             lr=hyperparams['lr'])

with experiment.train():                  
    for epoch in tqdm(range(hyperparams['num_epochs_ex']), desc='Epoch'):
        model.train()
        for x, y in tqdm(new_tr_loader, desc='Training'):
            x = normalize_tensor_wav(x)
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            train_loss_list.append(loss.item())
            loss_counter += 1
            if loss_counter % loss_cycle == 0:
                print()
                print('Average running loss:', sum(train_loss_list[-loss_cycle:]) / loss_cycle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        predictions = []
        truths = []
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(initial_val_loader, desc='Validating'):
                x = normalize_tensor_wav(x)
                x, y = x.to(device), y.to(device)
                yhat = torch.argmax(model(x), 1)
                truths.extend(y.tolist())
                predictions.extend(yhat.tolist())
  
        acc = sum(np.array(truths)==np.array(predictions))/len(truths)
        print()
        print('Accuracy: ', acc)
        val_acc_list.append(acc)
        experiment.log_metric("Validation accuracy", acc, step=hyperparams['num_epochs']+hyperparams['num_epochs_ex']+epoch) 