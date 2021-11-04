import torch
from torch.utils.data import Dataset
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import librosa

import numpy as np
from tqdm import tqdm
import os

import pickle as pkl

"""
PANN: raw wavform, 32k or 16k, normalize?
"""

class TAU2019Dataset(Dataset):    
    def __init__(self, 
            path,
            split, 
            feature='waveform',
            sr=32000,
            nfft=1024,
            window_sz=1024,
            hop_sz=320,
            mel_bins=64,
            fmin=14000,
            fmax=50000,     
            mel_normalize=False,
            transform=None
            ):
        super().__init__()
        
        assert split in ['train', 'eval']
        assert feature in ['waveform', 'mel']

        self.path = path
        self.split = split
        self.feature = feature
        self.sr = sr
        self.nfft = nfft
        self.window_sz = window_sz
        self.hop_sz = hop_sz
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.mel_normalize = mel_normalize
        
        self.labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']
        
        self.tau2019_dict = pkl.load(open('src/tau2019_dict_v2.pkl', "rb" ))
        
        self.ids = self._get_ids()
        
    def _get_ids(self):
        
        
        if self.split == 'train':
            ids = range(10105)
        else:
            ids = range(4295)
        
        ids = np.array(ids)
        
        if self.split == 'train':
            np.random.seed(0)
            np.random.shuffle(ids)
            
        return ids

    def _2D_normalize(self, spectrogram):
        return spectrogram / np.max(spectrogram)
    
    def _pad_or_remove(self, audio):
        length = len(audio)
        if length == self.sr*10:
            return audio
        elif length > self.sr*10:
            return audio[:self.sr*10]
        else:
            delta = self.sr*10 - length
            return np.pad(audio, (0, delta), 'constant')


    def __len__(self):
        if self.split == 'train':
            return 10105
        elif self.split == 'eval':
            return 4295

    def __getitem__(self, idx):
        audio_idx = self.ids[idx]
        
        if self.split == 'train':
            audio_basename = self.tau2019_dict['train'][audio_idx]
        else:
            audio_basename = self.tau2019_dict['eval'][audio_idx]
            
        audio_label = None
            
        for i, label in enumerate(self.labels):
            if label == 'metro':
                if label in audio_basename and 'station' not in audio_basename:
                    audio_label = i
            else:
                if label in audio_basename:
                    audio_label = i
            
        audio_file = os.path.join(self.path, audio_basename)
        
        audio, _ = librosa.core.load(audio_file, sr=self.sr, mono=True)
        
        audio = self._pad_or_remove(audio)
        
        audio = torch.tensor(audio)
        
        if self.feature == 'waveform':
            return audio, audio_label

        elif self.feature == 'mel':
            spectrogram_extractor = Spectrogram(n_fft=self.nfft, hop_length=self.hop_sz, 
                                                win_length=self.window_sz, window='hann', center=True, pad_mode='reflect', 
                                                freeze_parameters=True)

            logmel_extractor = LogmelFilterBank(sr=self.sr, n_fft=self.nfft, 
                                                n_mels=self.mel_bins, fmin=self.fmin, fmax=self.fmax, ref=1.0, amin=1e-10, top_db=None, 
                                                freeze_parameters=True)
            
            audio = spectrogram_extractor(audio)
            audio = logmel_extractor(audio)
            
            return audio, audio_label
