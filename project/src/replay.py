import torch
from torch.utils.data import Dataset, ConcatDataset, random_split, Subset
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os

def classwise_accuracy(predictions, truths, n_class, labels):
    '''
    @param predictions: N-dim numpy array
    @param truths: N-dim numpy array
    return accuracy averaged over classes, list of accuracy of each class
    '''

    acc = sum(predictions==truths)/len(truths)
    matrix = confusion_matrix(truths, predictions, labels=range(n_class))

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_class = matrix.diagonal()/matrix.sum(axis=1)
    acc_class = acc_class[labels]
        
    return acc, acc_class
    
    
class Replay(Dataset):
    def __init__(self,
                initial_tr,
                initial_classes, 
                keep_num,  # number of samples kept for each class
                ):
        super().__init__()
        self.num_per_class = keep_num
        self.audio_all_classes = {}
        self.mean_all_classes = {}  # Running means
        self.mean_counters = {}  # Use for running means
        self.classes = initial_classes.copy()
        self.num_class = len(self.classes)
        
        for label in initial_classes:
            self.audio_all_classes[label] = []
            self.mean_counters[label] = 0

        for i in tqdm(range(len(initial_tr)), desc='Replay::__init__: Loading audio data'):
            audio, label = initial_tr[i]
            self.audio_all_classes[label].append(audio)
            
        for label in self.audio_all_classes:
            self.audio_all_classes[label] = torch.Tensor(np.stack(self.audio_all_classes[label]))
            class_mean = torch.mean(self.audio_all_classes[label], dim=0)
            self.mean_all_classes[label] = class_mean
            self.mean_counters[label] = len(self.audio_all_classes[label])
            
            # Sort audios by L2 distance to the class mean
            dists = torch.norm(
                self.audio_all_classes[label] - class_mean, dim=-1
            )
            indices = torch.argsort(dists)
            self.audio_all_classes[label] = self.audio_all_classes[label][indices]
            
            # Only keep the top keep_num audios for each class
            self.audio_all_classes[label] = self.audio_all_classes[label][0:self.num_per_class]
                   
    def __len__(self):
        return self.num_class * self.num_per_class
    
    def __getitem__(self, idx):
        c = self.classes[idx//self.num_per_class]
        i = idx % self.num_per_class
        return self.audio_all_classes[c][i], c
         
    def update(self, 
              exposure,  # UrbanSoundExposure object
              label  # Inferred of a seen class or pseudo label of a unseen class
              ):
        
        if label not in self.classes:
            print('Insert a new class to the replay memory...')
            self.classes.append(label)
            self.num_class += 1
            self.audio_all_classes[label] = []
            for i in tqdm(range(len(exposure)), desc='Replay::update: Loading audio data'):
                audio, _ = exposure[i]
                self.audio_all_classes[label].append(audio)
            
            self.audio_all_classes[label] = torch.Tensor(np.stack(self.audio_all_classes[label]))
            class_mean = torch.mean(self.audio_all_classes[label], dim=0)
            self.mean_all_classes[label] = class_mean
            self.mean_counters[label] = len(self.audio_all_classes[label])
            
            # Sort audios by L2 distance to the class mean
            dists = torch.norm(
                self.audio_all_classes[label] - class_mean, dim=-1
            )
            indices = torch.argsort(dists)
            self.audio_all_classes[label] = self.audio_all_classes[label][indices]
            
            # Only keep the top keep_num audios for each class
            self.audio_all_classes[label] = self.audio_all_classes[label][0:self.num_per_class] 
            
        else:
            print('Update an existing class in the replay memory...')
            exposure_audios = []
            for i in tqdm(range(len(exposure)), desc='Replay::update: Loading audio data'):
                audio, _ = exposure[i]
                exposure_audios.append(audio)
                
            exposure_audios = torch.Tensor(np.stack(exposure_audios))
            
            old_class_count = self.mean_counters[label]
            old_class_mean = self.mean_all_classes[label]
            old_class_total = old_class_count * old_class_mean
            
            new_class_count = len(exposure_audios)
            new_class_mean = torch.mean(exposure_audios, dim=0)
            new_class_total = new_class_count * new_class_mean
            
            class_mean = (old_class_total + new_class_total) / (old_class_count + new_class_count)    
            self.mean_counters[label] = old_class_count + new_class_count

            self.audio_all_classes[label] = torch.cat(
                                                [self.audio_all_classes[label], exposure_audios],
                                                dim=0
                                            )
            
            # Sort audios by L2 distance to the class mean
            dists = torch.norm(
                self.audio_all_classes[label] - class_mean, dim=-1
            )
            indices = torch.argsort(dists)
            self.audio_all_classes[label] = self.audio_all_classes[label][indices]
            
            # Only keep the top keep_num audios for each class
            self.audio_all_classes[label] = self.audio_all_classes[label][0:self.num_per_class]  
            
            

class ReplayExposureBlender(Dataset):
    def __init__(self, 
            old, 
            new,
            old_labels,
            label=None,
            resize=None,
            transform=None,
            target_transform=None,
            transforms=None,
    ):
        super().__init__()
        assert len(old_labels) < 10
        
        if resize:
            down_idx = np.array([],dtype=int)
            class_sz = len(old) // len(old_labels)
            for i in range(len(old_labels)):
                class_idx = np.arange(class_sz) + i*class_sz
                np.random.shuffle(class_idx)
                down_idx = np.concatenate((down_idx, class_idx[:resize]))
            
            down_old = Subset(old, down_idx)

            self.old_num = len(down_old)
            self.dataset = ConcatDataset((down_old, new))
        else:
            self.old_num = len(old)
            self.dataset = ConcatDataset((old, new))
            
        self.new_num = len(new)   
        
        if label != None:
            self.pseudo_label = label
                
        else: 
            # Assign a new label to the exposure no matter seen or not
            for i in range(10):
                if i not in old_labels:
                    self.pseudo_label = i
                    break
            
    def __len__(self):
        return self.old_num + self.new_num
    
    def __getitem__(self, idx):
        if idx < self.old_num:
            #return torch.tensor(self.dataset[idx][0]), self.dataset[idx][1]
            return torch.tensor(self.dataset[idx][0]), self.dataset[idx][1]
        else:
            #return torch.tensor(self.dataset[idx][0]), self.pseudo_label
            return torch.tensor(self.dataset[idx][0]), self.pseudo_label

    def update_label(self, label):
        self.pseudo_label = label
        
if __name__ == '__main__':
    predictions = np.array([9, 2, 5, 3, 4, 0, 7, 2, 9, 3, 4, 2])
    truths = np.array([9, 2, 3, 3, 2, 0, 9, 2, 9, 3, 0, 2])
    
    print(classwise_accuracy(predictions, truths, 10, [9, 0, 2, 3]))