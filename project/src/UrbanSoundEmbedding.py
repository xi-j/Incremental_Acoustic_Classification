import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from tqdm import tqdm
import os
import random
import csv
import wav2clip

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
        self.num_samples = 64000  # hardcoded
        
        #self.encoder = wav2clip.get_model()
        
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
        
        
        audio = self._pad_or_remove(audio)
        
        #embeddings = wav2clip.embed_audio(audio, self.encoder)
        #print(audio)
        
        return audio, label
