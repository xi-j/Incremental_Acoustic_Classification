import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
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
    

class ReplayExposureBlender(Dataset):
    def __init__(self, 
            old, 
            new,
            old_labels,
            new_label,  # This is the truth label
            transform=None,
            target_transform=None,
            transforms=None,
    ):
        super().__init__()
        assert len(old_labels) < 10
        
        self.old_num = len(old)
        self.new_num = len(new)
        
        self.true_label = new_label
        # Assign a new label to the exposure no matter seen or not
        for i in range(10):
            if i not in old_labels:
                self.fake_label = i
                break
         
        self.dataset = ConcatDataset((old, new))
        
        
    def __len__(self):
        return self.old_num + self.new_num
    
    def __getitem__(self, idx):
        if idx < self.old_num:
            return torch.Tensor(self.dataset[idx][0]), self.dataset[idx][1]
        else:
            return torch.Tensor(self.dataset[idx][0]), self.fake_label

        
if __name__ == '__main__':
    predictions = np.array([9, 2, 5, 3, 4, 0, 7, 2, 9, 3, 4, 2])
    truths = np.array([9, 2, 3, 3, 2, 0, 9, 2, 9, 3, 0, 2])
    
    print(classwise_accuracy(predictions, truths, 10, [9, 0, 2, 3]))