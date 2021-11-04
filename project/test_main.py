import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.ESC10 import ESC10Dataset, ESC10ExposureGenerator
from src.classifier import SimpleClassifier
from src.CNNs import Cnn6

from src.TAU2019 import TAU2019Dataset

esc10_tr = ESC10Dataset('ESC-10', 'train', feature='mel', transform=ToTensor())
esc10_tr_loader = DataLoader(esc10_tr, batch_size=4, shuffle=True, num_workers=1)

""""
for x, labels in esc10_tr_loader:
    print(x.shape)
    print(labels)
    break
    
"""

tau_dev_path='/mnt/data/DCASE2019/Task1/TAU-urban-acoustic-scenes-2019-development/audio/'

tau2019_train = TAU2019Dataset(path=tau_dev_path,split='train', transform=ToTensor())
tau2019_train_loader = DataLoader(tau2019_train, batch_size=16, shuffle=True, num_workers=4)

tau2019_eval = TAU2019Dataset(path=tau_dev_path,split='eval', transform=ToTensor())
tau2019_eval_loader = DataLoader(tau2019_eval, batch_size=16, shuffle=False, num_workers=4)


    

device = torch.device('cuda:2')
model = Cnn6(32000, 1024, 320, 64, 14000, 50000, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

loss_list = []
loss_counter = 0

max_acc = 0

predictions = []
truths = []
    
with torch.no_grad():
    for x, labels in tau2019_eval_loader:
        x, labels = x.to(device), labels.to(device)
        predicts = torch.argmax(model(x),1)
            
        truths.extend(labels.tolist())
        predictions.extend(predicts.tolist())
            
acc = sum(np.array(truths)==np.array(predictions))/len(truths)
        
print('Accuracy: ', acc)

for epoch in range(10):
    print(f'Epoch {epoch}')
    for x, labels in tau2019_train_loader:
        x, labels = x.to(device), labels.to(device)
        predicts = model(x)
        #print(predicts)
        #print(labels)
        loss = criterion(predicts, labels)

        loss_list.append(loss.item())
        loss_counter += 1
        if loss_counter % 50 == 0:
            print('Average running loss:', sum(loss_list[-10:]) / 10)
        #print(loss)
        #print(accuracy(labels, predicts))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    predictions = []
    truths = []
    
    with torch.no_grad():
        for x, labels in tau2019_eval_loader:
            x, labels = x.to(device), labels.to(device)
            predicts = torch.argmax(model(x),1)
            
            truths.extend(labels.tolist())
            predictions.extend(predicts.tolist())
            
    acc = sum(np.array(truths)==np.array(predictions))/len(truths)
        
    print('Accuracy: ', acc)
    
    if acc > max_acc:
        max_acc = acc
            
        PATH = 'cpkts/model' + str(epoch) + '.pth'
        torch.save(model.state_dict(), PATH)


