import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from tqdm import tqdm
import os
import random
import csv

class TAUDataset(Dataset):
    '''
    Remember to use torch.utils.data.random_split to split TAUDataset to train and test
    '''
    
    classes = [
        'airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square',
        'street_traffic', 'tram', 'bus', 'metro', 'park' 
    ]
    class_to_int = {
        'airport':0, 'shopping_mall':1, 'metro_station':2, 
        'street_pedestrian':3, 'public_square':4, 'street_traffic':5, 
        'tram':6, 'bus':7, 'metro':8, 'park':9
    }
    
    
    def __init__(self, 
            path, 
            sr=48000, # native sampling rate
            transform=None,
            target_transform=None,
            transforms=None,
            ):
        super().__init__()
        self.sr = sr
        self.audio_path = os.path.join(path, 'audio')
        self.num_samples = sr * 10  # hardcoded
        
        # files by classes
        self.audio_files = []
        # all files
        self.audio_all_files = []
        for i in range(10):
            self.audio_files.append([])
        
        for filename in os.listdir(self.audio_path):
            if filename.endswith(".wav"):
                for label in self.classes:
                    if label in filename:
                        if label != 'metro':
                            self.audio_all_files.append(
                                (os.path.join(self.audio_path, filename), self.class_to_int[label])
                            )
                            self.audio_files[self.class_to_int[label]]\
                            .append(os.path.join(self.audio_path, filename))
                        elif 'metro_station' in filename:
                            continue        
                        else:
                            self.audio_all_files.append(
                                (os.path.join(self.audio_path, filename), self.class_to_int['metro'])
                            )
                            self.audio_files[self.class_to_int[label]]\
                            .append(os.path.join(self.audio_path, filename))    
                        

    def _pad_or_remove(self, audio):
        length = len(audio)
        if length == self.num_samples:
            return audio
        elif length > self.num_samples:
            return audio[:self.num_samples]
        else:
            delta = self.num_samples - length
            return np.pad(audio, (0, delta), 'constant')
            
    def __len__(self):
        return len(self.audio_all_files)

    def __getitem__(self, idx: int):        
        file = self.audio_all_files[idx][0]
        label = self.audio_all_files[idx][1]
        audio, sr = librosa.load(file, sr=self.sr)
        return torch.Tensor(self._pad_or_remove(audio)), label

class TAUTestset(Dataset):    
    classes = [
        'airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square',
        'street_traffic', 'tram', 'bus', 'metro', 'park' 
    ]
    class_to_int = {
        'airport':0, 'shopping_mall':1, 'metro_station':2, 
        'street_pedestrian':3, 'public_square':4, 'street_traffic':5, 
        'tram':6, 'bus':7, 'metro':8, 'park':9
    }
    
    
    def __init__(self, 
            test_files, # Dict of list of test audio files
            sr=48000, # native sampling rate
            transform=None,
            target_transform=None,
            transforms=None,
            ):
        super().__init__()
        self.sr = sr
        self.test_files = test_files
        self.num_samples = sr * 10  # hardcoded
        
        # all test files
        self.audio_all_files = []
        for i in range(10):
            for file in test_files[i]:
                self.audio_all_files.append((file, i))
                            

    def _pad_or_remove(self, audio):
        length = len(audio)
        if length == self.num_samples:
            return audio
        elif length > self.num_samples:
            return audio[:self.num_samples]
        else:
            delta = self.num_samples - length
            return np.pad(audio, (0, delta), 'constant')
            
    def __len__(self):
        return len(self.audio_all_files)

    def __getitem__(self, idx: int):        
        file = self.audio_all_files[idx][0]
        label = self.audio_all_files[idx][1]
        audio, sr = librosa.load(file, sr=self.sr)
        return torch.Tensor(self._pad_or_remove(audio)), label
    
    
class TAUExposureGenerator(Dataset):
    classes = [
        'airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square',
        'street_traffic', 'tram', 'bus', 'metro', 'park' 
    ]
    class_to_int = {
        'airport':0, 'shopping_mall':1, 'metro_station':2, 
        'street_pedestrian':3, 'public_square':4, 'street_traffic':5, 
        'tram':6, 'bus':7, 'metro':8, 'park':9
    }    
    class TAUInitialClasses(Dataset):
        def __init__(self,
                    files,
                    labels,
                    sr
                    ):
            super().__init__()

            self.files = files
            self.labels = labels
            self.sr = sr
            self.num_samples = sr * 10  # hardcoded

        def _pad_or_remove(self, audio):
            length = len(audio)
            if length == self.num_samples:
                return audio
            elif length > self.num_samples:
                return audio[:self.num_samples]
            else:
                delta = self.num_samples - length
                return np.pad(audio, (0, delta), 'constant')

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file = self.files[idx]
            label = self.labels[idx]
            audio, sr = librosa.load(file, self.sr)

            return self._pad_or_remove(audio), label

    class TAUExposure(Dataset):
        def __init__(self, 
                    exposure_files,
                    exposure_label,
                    sr
                    ):
            super().__init__()
            self.exposure_files = exposure_files
            self.exposure_label = exposure_label
            self.sr = sr
            self.num_samples = sr * 10 # hardcoded

        def _pad_or_remove(self, audio):
            length = len(audio)
            if length == self.num_samples:
                return audio
            elif length > self.num_samples:
                return audio[:self.num_samples]
            else:
                delta = self.num_samples - length
                return np.pad(audio, (0, delta), 'constant')

        def __len__(self):
            return len(self.exposure_files)

        def __getitem__(self, idx: int):
            file = self.exposure_files[idx]
            audio, sr = librosa.load(file, self.sr)
            
            return torch.Tensor(self._pad_or_remove(audio)), self.exposure_label
        
    def __init__(self, 
            path,
            test_size=240, # number of test audios per class reserved
            sr=48000, # native sampling frequency
            exposure_size=300,
            exposure_val_size=50, 
            initial_K=4,
            transform=None,
            target_transform=None,
            transforms=None,
            ):
        super().__init__()
        assert exposure_size > 0
        assert exposure_val_size > 0 and exposure_val_size < exposure_size
        
        self.sr = sr
        self.audio_path = os.path.join(path, 'audio')
        self.test_size = test_size
        self.num_samples = sr * 10  # hardcoded
        self.exposure_size = exposure_size  # number of samples for each exposure
        self.exposure_val_size = exposure_val_size  # number of validation samples in each exposure
        self.exposure_per_class = []  # number of exposures per class
        self.initial_K = initial_K
        self.initial_classes = random.sample(range(0, 10), self.initial_K)
            
        # files by classes
        self.audio_files = []
        # reserved test files
        self.test_files = {}
        self.train_files = {}

        for i in range(10):
            self.audio_files.append([])
            self.test_files[i] = 0
            self.train_files[i] = 0

        for filename in os.listdir(self.audio_path):
            if filename.endswith(".wav"):
                for label in self.classes:
                    if label in filename:
                        if label != 'metro':
                            self.audio_files[self.class_to_int[label]]\
                            .append(os.path.join(self.audio_path, filename))
                        elif 'metro_station' in filename:
                            continue        
                        else:
                            self.audio_files[self.class_to_int[label]]\
                            .append(os.path.join(self.audio_path, filename))
    
        # Shuffle all files and index exposures
        self.exposure_remain_idx = []
        for i in range(10):
            random.seed(i)
            random.shuffle(self.audio_files[i])
            self.audio_files[i] = self.audio_files[i][0:-self.test_size]
            self.test_files[i] = self.audio_files[i][-self.test_size:]
            self.train_files[i] = self.audio_files[i][0:-self.test_size]
                        
            exposure_num = len(self.audio_files[i])//self.exposure_size
            self.exposure_per_class.append(exposure_num)
            for j in range(exposure_num):
                self.exposure_remain_idx.append((i, j))
           
        random.shuffle(self.exposure_remain_idx)
        self.exposure_max = len(self.exposure_remain_idx)
        
    def get_test_set(self):
        return TAUTestset(self.test_files, self.sr)

    def get_train_set(self):
        return TAUTestset(self.train_files, self.sr)
        
    def get_initial_set(self, initial_classes=None):
        '''
        Return an TAUExposureGenerator.TAUInitialClasses object, 
        a Dataset that contains one exposure for each one of the initial classes
        '''
        if initial_classes != None:
            assert len(initial_classes) == self.initial_K
            self.initial_classes = initial_classes

        val_files = []
        train_files = []
        val_labels = []
        train_labels = []
        for c in self.initial_classes:
            temp = list(zip(self.audio_files[c][0:self.exposure_size], [c] * self.exposure_size))   
            self.exposure_remain_idx.remove((c, 0))
            
            random.shuffle(temp)
            files, labels = zip(*temp)  
            
            val_files += files[:self.exposure_val_size]
            train_files += files[self.exposure_val_size:]
            val_labels += labels[:self.exposure_val_size]
            train_labels += labels[self.exposure_val_size:]        
        
        
        return(
            TAUExposureGenerator.TAUInitialClasses(
                files=train_files, labels=train_labels, sr=self.sr
            ),
            TAUExposureGenerator.TAUInitialClasses(
                files=val_files, labels=val_labels, sr=self.sr
            ),
            self.initial_classes
        )

    def __len__(self):
        return self.exposure_max - self.initial_K

    def __getitem__(self, idx: int):
        '''
        Randomly return an TAUExposureGenerator.TAUExposure object, 
        a Dataset that contains self.exposure_size audios of the same class
        '''

        c, ei = self.exposure_remain_idx.pop(0)

        files = self.audio_files[c][ei*self.exposure_size:(ei+1)*self.exposure_size]
        random.shuffle(files)
        
        val_files = files[:self.exposure_val_size]
        train_files = files[self.exposure_val_size:]
        
        return (
            TAUExposureGenerator.TAUExposure(
                exposure_files=train_files, 
                exposure_label=c, 
                sr=self.sr
            ),
            TAUExposureGenerator.TAUExposure(
                exposure_files=val_files, 
                exposure_label=c, 
                sr=self.sr
            ),
            c
        )
