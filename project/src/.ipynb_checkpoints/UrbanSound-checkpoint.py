import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from tqdm import tqdm
import os
import random
import csv

class UrbanSoundDataset(Dataset):
    def __init__(self, 
            path, 
            folders,
            sr,
            transform=None,
            target_transform=None,
            transforms=None,
            ):
        super().__init__()
        self.folders = folders
        self.sr = sr
        self.audio_path = os.path.join(path, 'audio')
        self.meta_path = os.path.join(path, 'metadata', 'UrbanSound8K.csv')
        self.num_samples = sr * 4  # hardcoded
        
        # files by classes
        self.audio_files = {}
        # all files
        self.audio_all_files = []
        for i in range(10):
            self.audio_files[i] = []
        
        with open(self.meta_path, newline='') as meta:
            reader = csv.reader(meta, delimiter=' ', quotechar='|')
            next(reader, None)
            for row in reader:
                row = row[0].split(',')
                folder = int(row[5])
                if folder not in folders:
                    continue
                label = int(row[6])
                file = row[0]
                self.audio_files[label].append(
                    os.path.join(self.audio_path, 'fold'+str(folder), file))
                self.audio_all_files.append((
                    os.path.join(self.audio_path, 'fold'+str(folder), file),
                    label)
                )

    def _pad_or_remove(self, audio):
        length = len(audio)
        if length == self.num_samples:
            return audio
        elif length > self.num_samples:
            return audio[:self.num_samples]
        else:
            delta = self.num_samples - length
            delta = self.num_samples - length
            return np.pad(audio, (0, delta), 'constant')
            
    def __len__(self):
        return len(self.audio_all_files)

    def __getitem__(self, idx: int):        
        file = self.audio_all_files[idx][0]
        label = self.audio_all_files[idx][1]
        audio, sr = librosa.load(file, self.sr)

        return torch.Tensor(self._pad_or_remove(audio)), label

class UrbanSoundExposureGenerator(Dataset):
    
    class UrbanSoundInitialClasses(Dataset):
        def __init__(self,
                    files,
                    labels,
                    sr
                    ):
            super().__init__()

            self.files = files
            self.labels = labels
            self.sr = sr
            self.num_samples = sr * 4  # hardcoded

        def _pad_or_remove(self, audio):
            length = len(audio)
            if length == self.num_samples:
                return audio
            elif length > self.num_samples:
                return audio[:self.num_samples]
            else:
                delta = self.num_samples - length
                delta = self.num_samples - length
                return np.pad(audio, (0, delta), 'constant')

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file = self.files[idx]
            label = self.labels[idx]
            audio, sr = librosa.load(file, self.sr)

            return self._pad_or_remove(audio), label

    class UrbanSoundExposure(Dataset):
        def __init__(self, 
                    exposure_files,
                    exposure_label,
                    sr
                    ):
            super().__init__()
            self.exposure_files = exposure_files
            self.exposure_label = exposure_label
            self.sr = sr
            self.num_samples = sr * 4  # hardcoded

        def _pad_or_remove(self, audio):
            length = len(audio)
            if length == self.num_samples:
                return audio
            elif length > self.num_samples:
                return audio[:self.num_samples]
            else:
                delta = self.num_samples - length
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
            folders,
            sr,
            exposure_size=300,
            exposure_val_size=50, 
            initial_K=4,
            transform=None,
            target_transform=None,
            transforms=None,
            ):
        super().__init__()
        assert exposure_size > 0 and exposure_size < 900
        assert exposure_val_size > 0 and exposure_val_size < exposure_size
        
        self.folders = folders
        self.sr = sr
        self.audio_path = os.path.join(path, 'audio')
        self.meta_path = os.path.join(path, 'metadata', 'UrbanSound8K.csv')
        self.num_samples = sr * 4  # hardcoded
        self.exposure_size = exposure_size  # number of samples for each exposure
        self.exposure_val_size = exposure_val_size  # number of validation samples in each exposure
        self.exposure_per_class = []  # number of exposures per class
        self.initial_K = initial_K
        self.initial_classes = random.sample(range(0, 10), self.initial_K)
         
        # All files by classes
        self.audio_files = {}
        for i in range(10):
            self.audio_files[i] = []
        with open(self.meta_path, newline='') as meta:
            reader = csv.reader(meta, delimiter=' ', quotechar='|')
            next(reader, None)
            for row in reader:
                row = row[0].split(',')
                folder = int(row[5])
                if folder not in folders:
                    continue
                label = int(row[6])
                file = row[0]
                self.audio_files[label].append(
                    os.path.join(self.audio_path, 'fold'+str(folder), file))
    
        # Shuffle all files and index exposures
        self.exposure_remain_idx = []
        for i in range(10):
            random.shuffle(self.audio_files[i])
            exposure_num = len(self.audio_files[i])//self.exposure_size
            self.exposure_per_class.append(exposure_num)
            for j in range(exposure_num):
                self.exposure_remain_idx.append((i, j))
           
        random.shuffle(self.exposure_remain_idx)
        self.exposure_max = len(self.exposure_remain_idx)
        
    def get_initial_set(self, initial_classes=None):
        '''
        Return an UrbanSoundExposureGenerator.UrbanSoundInitialClasses object, 
        a Dataset that contains one exposure for each one of the initial classes
        '''
        if initial_classes != None:
            assert len(initial_classes) == self.initial_K
            self.initial_classes = initial_classes

        files = []
        labels = []
        for c in self.initial_classes:
            files += self.audio_files[c][0:self.exposure_size]
            labels += [c] * self.exposure_size
            self.exposure_remain_idx.remove((c, 0))
          
        temp = list(zip(files, labels))
        random.shuffle(temp)
        files, labels = zip(*temp)

        val_files = files[:self.initial_K*self.exposure_val_size]
        train_files = files[self.initial_K*self.exposure_val_size:]
        val_labels = labels[:self.initial_K*self.exposure_val_size]
        train_labels = labels[self.initial_K*self.exposure_val_size:]
        
        return(
            UrbanSoundExposureGenerator.UrbanSoundInitialClasses(
                files=train_files, labels=train_labels, sr=self.sr
            ),
            UrbanSoundExposureGenerator.UrbanSoundInitialClasses(
                files=val_files, labels=val_labels, sr=self.sr
            ),
            self.initial_classes
        )

    def __len__(self):
        return self.exposure_max - self.initial_K

    def __getitem__(self, idx: int):
        '''
        Randomly return an UrbanSoundExposureGenerator.UrbanSoundExposure object, 
        a Dataset that contains self.exposure_size audios of the same class
        '''

        c, ei = self.exposure_remain_idx.pop(0)

        files = self.audio_files[c][ei*self.exposure_size:(ei+1)*self.exposure_size]
        random.shuffle(files)
        
        val_files = files[:self.exposure_val_size]
        train_files = files[self.exposure_val_size:]
        
        return (
            UrbanSoundExposureGenerator.UrbanSoundExposure(
                exposure_files=train_files, 
                exposure_label=c, 
                sr=self.sr
            ),
            UrbanSoundExposureGenerator.UrbanSoundExposure(
                exposure_files=val_files, 
                exposure_label=c, 
                sr=self.sr
            ),
            c
        )
