import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
import os

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
