import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
from librosa.feature import melspectrogram
from tqdm import tqdm
import os
import random

class ESC10Dataset(Dataset):
    int_to_label = {0:'fire_cracking', 1:'dog_bark', 2:'rain', 3:'sea_waves', 4:'baby_cry',
                    5:'clock_tick', 6:'person_sneeze', 7:'helicopter', 8:'chainsaw', 9:'rooster'
                    }

    def __init__(self, 
            path, 
            split, 
            feature='waveform',
            mel_normalize=True,
            transform=None,
            target_transform=None,
            transforms=None,
            ):
        super().__init__()
        assert split in ['train', 'val']
        assert feature in ['waveform', 'mel']

        self.path = os.path.join(path, split)
        self.split = split
        self.feature = feature
        self.mel_normalize = mel_normalize
        self.num_samples = 220500  # hardcoded

        self.audio_files = {}

        for i in range(10):
            self.audio_files[i] = []

        for i, subdir in enumerate(
                    ['010 - Fire crackling', '001 - Dog bark', '002 - Rain', '003 - Sea waves', 
                    '004 - Baby cry', '005 - Clock tick', '006 - Person sneeze', '007 - Helicopter', 
                    '008 - Chainsaw', '009 - Rooster'
                    ]
                    ):
            for f in os.listdir(os.path.join(self.path, subdir)):
                ext = os.path.splitext(f)[-1].lower()
                if ext != '.ogg':
                    continue
                else:
                    self.audio_files[i].append(os.path.join(self.path, subdir, f))

    def _2D_normalize(self, spectrogram):
        return spectrogram / np.max(spectrogram)

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
        if self.split == 'train':
            return 300
        elif self.split == 'val':
            return 100

    def __getitem__(self, idx: int):
        if self.split == 'train':
            label = idx // 30
            audio_idx = idx % 30
        elif self.split == 'val':
            label = idx // 10
            audio_idx = idx % 10
        
        filename = self.audio_files[label][audio_idx]
        audio, sr = sf.read(filename)

        if self.feature == 'waveform':
            audio = self._pad_or_remove(audio)
            return torch.tensor(audio), label

        elif self.feature == 'mel':
            audio = self._pad_or_remove(audio)
            spectrogram = melspectrogram(y=audio, sr=sr, power=1)
            if self.mel_normalize:
                spectrogram = self._2D_normalize(spectrogram)
            return torch.FloatTensor(spectrogram), label

class ESC10ExposureGenerator(Dataset):
    int_to_subdir = {0:'010 - Fire crackling', 1:'001 - Dog bark', 2:'002 - Rain', 3:'003 - Sea waves', 4:'004 - Baby cry',
                    5:'005 - Clock tick', 6:'006 - Person sneeze', 7:'007 - Helicopter', 8:'008 - Chainsaw', 9:'009 - Rooster'
                    }

    class ESC10InitialClasses(Dataset):
        def __init__(self,
                    files,
                    labels,
                    feature='waveform',
                    mel_normalize=True,
                    ):
            super().__init__()
            assert feature in ['waveform', 'mel']

            self.files = files
            self.labels = labels
            self.feature = feature
            self.num_samples = 220500  # hardcoded
            self.mel_normalize = mel_normalize

        def _2D_normalize(self, spectrogram):
            return spectrogram / np.max(spectrogram)

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
            filename = self.files[idx]
            label = self.labels[idx]
            audio, sr = sf.read(filename) 

            if self.feature == 'waveform':
                audio = self._pad_or_remove(audio)
                return torch.tensor(audio), label

            elif self.feature == 'mel':
                audio = self._pad_or_remove(audio)
                spectrogram = melspectrogram(y=audio, sr=sr, power=1)
                if self.mel_normalize:
                    spectrogram = self._2D_normalize(spectrogram)
                return torch.FloatTensor(spectrogram), label

    class ESC10Exposure(Dataset):
        def __init__(self, 
                    exposure_files,
                    exposure_label,
                    feature='waveform',
                    mel_normalize=True,
                    ):
            super().__init__()
            assert feature in ['waveform', 'mel']

            self.exposure_files = exposure_files
            self.exposure_label = exposure_label
            self.feature = feature
            self.num_samples = 220500  # hardcoded
            self.mel_normalize = mel_normalize

        def _2D_normalize(self, spectrogram):
            return spectrogram / np.max(spectrogram)

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
            filename =  self.exposure_files[idx]
            audio, sr = sf.read(filename)

            if self.feature == 'waveform':
                audio = self._pad_or_remove(audio)
                return torch.tensor(audio), self.exposure_label
            elif self.feature == 'mel':
                audio = self._pad_or_remove(audio)
                spectrogram = melspectrogram(y=audio, sr=sr, power=1)
                if self.mel_normalize:
                    spectrogram = self._2D_normalize(spectrogram)

            return torch.FloatTensor(spectrogram), self.exposure_label

    def __init__(self, 
            path, 
            feature='waveform',
            mel_normalize=True,
            transform=None,
            target_transform=None,
            transforms=None,
            exposure_size=10,
            initial_K=4
            ):
        super().__init__()
        assert feature in ['waveform', 'mel']

        self.split = 'train'
        self.path = os.path.join(path, 'train')
        self.feature = feature
        self.mel_normalize = mel_normalize
        self.num_samples = 220500  # hardcoded
        self.files_per_class = 30  # hardcoded
        self.exposure_size = exposure_size
        self.exposure_per_class = self.files_per_class // exposure_size

        self.initial_K = initial_K

        # These are the initial K classes that the model trains on 
        self.initial_classes = random.sample(range(0, 10), self.initial_K)

        # audio_files[i] is the list of all exposures of class i (length exposure_per_class)
        # audio_files[i][j] is the list of all audios of exposure j of class i (length exposure_size)
        self.audio_files = {}
        # This contains indices(class_idx, exposure_idx) of exposures that are not returned yet
        self.exposure_remain_idx = []#[self.exposure_per_class] * 10

        for i in range(10):
            self.audio_files[i] = [[] for _ in range(self.exposure_per_class)]
            for j in range(self.exposure_per_class):
                self.exposure_remain_idx.append((i, j))

        for i, subdir in enumerate(
                    ['010 - Fire crackling', '001 - Dog bark', '002 - Rain', '003 - Sea waves', 
                    '004 - Baby cry', '005 - Clock tick', '006 - Person sneeze', '007 - Helicopter', 
                    '008 - Chainsaw', '009 - Rooster'
                    ]
                    ):
            fi = 0
            for f in os.listdir(os.path.join(self.path, subdir)):
                ext = os.path.splitext(f)[-1].lower()
                if ext != '.ogg':
                    continue
                else:
                    self.audio_files[i][fi//self.exposure_size].append(os.path.join(self.path, subdir, f))
                    fi += 1

        # Shuffle exposures 
        random.shuffle(self.exposure_remain_idx)
        self.exposure_max = len(self.exposure_remain_idx) - self.initial_K


    def get_initial_set(self, initial_classes=None):
        '''
        Return an ESC10ExposureGenerator.ESC10InitialClasses object, which is a Dataset
        that contains one exposure for each one of the initial classes
        '''

        if initial_classes != None:
            assert len(initial_classes) == self.initial_K
            self.initial_classes = initial_classes

        files = []
        labels = []
        for c in self.initial_classes:
            # randomly choose an exposure of the class
            r = random.randint(0, self.exposure_per_class-1)

            files += self.audio_files[c][r]
            labels += [c] * self.exposure_size
            
            self.exposure_remain_idx.remove((c, r))

        return ESC10ExposureGenerator.ESC10InitialClasses(files=files, labels=labels,
                        feature=self.feature, mel_normalize=self.mel_normalize)

    def __len__(self):
        return self.exposure_max

    def __getitem__(self, idx: int):
        '''
        Randomly return an ESC10ExposureGenerator.ESC10Exposure object, which is a Dataset
        that contains self.exposure_size audios
        '''

        c, ei = self.exposure_remain_idx.pop(0)
        return ESC10ExposureGenerator.ESC10Exposure(exposure_files=self.audio_files[c][ei], 
                                                    exposure_label=c,
                                                    feature=self.feature,
                                                    mel_normalize=self.mel_normalize
                                                    )




