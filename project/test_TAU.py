from random import expovariate
from src.TAU2019 import TAUDataset, TAUExposureGenerator
from torch.utils.data import DataLoader

dataset = TAUDataset(
    path='D:\Datasets\TAU',
    sr=16000
)

exposure_generator = TAUExposureGenerator(
    'D:\Datasets\TAU', 
    sr=16000, 
    exposure_size=300, 
    exposure_val_size=50, 
    initial_K=4
)

print(len(exposure_generator))

initial_tr, initial_val, seen_classes = exposure_generator.get_initial_set()

print(len(exposure_generator))

"""
exposure_tr_list = []
exposure_val_list = []
exposure_label_list = []
for i in range(len(exposure_generator)):
    exposure_tr, exposure_val, label  = exposure_generator[i]  
    exposure_tr_list.append(exposure_tr)
    exposure_val_list.append(exposure_val)
    exposure_label_list.append(label)

initial_tr_loader = DataLoader(initial_tr, batch_size=4, 
                               shuffle=True, num_workers=4)
initial_val_loader = DataLoader(initial_val, batch_size=4, 
                                shuffle=True, num_workers=4)
"""

train_set = exposure_generator.get_train_set()
test_set = exposure_generator.get_test_set()

for x,y in train_set:
    print(x)
    print(y)
    break