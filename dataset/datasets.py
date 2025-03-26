from torch.utils.data import Dataset, Subset, random_split
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import config
import random
import numpy as np
import soundata


class AudioDataset(Dataset):
    def __init__(self, root: str, args, download: bool = True, dataset_name: str = 'esc50'):
        self.root = os.path.join(root, dataset_name)
        self.dataset_name = dataset_name
        self.args = args
        '''
            Dataset Initialization using soundata
            https://github.com/soundata/soundata.git
        '''
        dataset = soundata.initialize(dataset_name=self.dataset_name, data_home=self.root)
        if download:
            dataset.download()
        self.ids = dataset.clip_ids
        self.clips = dataset.load_clips()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
class ESC50(AudioDataset):
    def __init__(self, root: str, args, download: bool = True, dataset_name: str = 'esc50'):
        super().__init__(root, args, download, dataset_name)

        print("Bulding Indexes")
        self.class_to_idx = {}
        self.categories, self.targets, self.fold_nums, self.audio_tensors = [], [], [], []
        for id in tqdm(self.ids, total=len(self.ids)):
            clip = self.clips[id]
            category, target, fold = clip.category, clip.target, clip.fold
            if self.class_to_idx.get(category) is None:
                self.class_to_idx[category] = target
            self.categories.append(category)
            self.targets.append(target)
            self.fold_nums.append(fold)
            self.audio_tensors.append(self.get_audio_embeddings(clip))
        
        self.audio_tensors = torch.stack(self.audio_tensors, dim=0)
        # eps = args.soft_epsilon
        # self.targets = (1-eps)*F.one_hot(torch.tensor(self.targets), num_classes=len(self.class_to_idx)).float() + eps/len(self.class_to_idx)
        # self.targets = self.targets.reshape(-1, len(self.class_to_idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # K-Fold Split
        self.k = len(set(self.fold_nums))
        self.folds = self.k_fold_split(self.k)


    def __getitem__(self, index):
        audio_tensor = self.audio_tensors[index]
        target = torch.tensor(self.targets[index])
        
        return audio_tensor, target

    def __len__(self):
        return len(self.audio_tensors)
    
    def get_audio_embeddings(self, clip):
        waveform, sample_rate = clip.audio
        waveform = waveform.reshape(-1)
        # waveform is shorter than predefined audio duration,
        # so waveform is extended
        if config.audio_duration*sample_rate >= waveform.shape[0]:
            repeat_factor = int(np.ceil((config.audio_duration*sample_rate) /
                                        waveform.shape[0]))
            # Repeat waveform by repeat_factor to match config.audio_duration
            waveform = waveform.repeat(repeat_factor)
            # remove excess part of waveform
            waveform = waveform[0:config.audio_duration*sample_rate]
        else:
            # waveform is longer than predefined audio duration,
            # so waveform is trimmed
            start_index = random.randrange(
                waveform.shape[0] - config.audio_duration*sample_rate)
            waveform = waveform[start_index:start_index +config.audio_duration*sample_rate]
        
        audio_tensor = torch.FloatTensor(waveform).reshape(-1)
        # audio_tensor = audio_tensor.reshape(1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)
        # audio_tensor = audio_tensor.reshape(1, -1)
        
        return audio_tensor

    def k_fold_split(self, k:int):
        folds = []
        self.fold_nums = np.array(self.fold_nums)
        for i in range(1, k+1):
            indices = np.where(self.fold_nums == i)[0]
            folds.append(Subset(self, indices))
        return folds

class UrbanSound8k(AudioDataset):
    def __init__(self, root: str, args, download: bool = True, dataset_name: str = 'esc50'):
        super().__init__(root)

        print("Bulding Indexes")
        self.class_to_idx = {}
        self.categories, self.targets, self.fold_nums, self.audio_tensors = [], [], [], []
        for id in tqdm(self.ids, total=len(self.ids)):
            clip = self.clips[id]
            category, target, fold = clip.class_label, clip.class_id, clip.fold
            if self.class_to_idx.get(category) is None:
                self.class_to_idx[category] = target
            self.categories.append(category)
            self.targets.append(target)
            self.fold_nums.append(fold)
            self.audio_tensors.append(self.get_audio_embeddings(clip))
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # K-Fold Split
        self.k = len(set(self.fold_nums))
        self.folds = self.k_fold_split(self.k)


    def __getitem__(self, index):
        audio_tensor = self.audio_tensors[index]
        target = self.targets[index]
        
        return audio_tensor, target

    def __len__(self):
        return len(self.audio_tensors)
    
    def get_audio_embeddings(self, clip):
        waveform, sample_rate = clip.audio
        waveform = waveform.reshape(-1)
        # waveform is shorter than predefined audio duration,
        # so waveform is extended
        if config.audio_duration*sample_rate >= waveform.shape[0]:
            repeat_factor = int(np.ceil((config.audio_duration*sample_rate) /
                                        waveform.shape[0]))
            # Repeat waveform by repeat_factor to match config.audio_duration
            waveform = waveform.repeat(repeat_factor)
            # remove excess part of waveform
            waveform = waveform[0:config.audio_duration*sample_rate]
        else:
            # waveform is longer than predefined audio duration,
            # so waveform is trimmed
            start_index = random.randrange(
                waveform.shape[0] - config.audio_duration*sample_rate)
            waveform = waveform[start_index:start_index +config.audio_duration*sample_rate]
        
        audio_tensor = torch.FloatTensor(waveform).reshape(-1)
        audio_tensor = audio_tensor.reshape(1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)
        audio_tensor = audio_tensor.reshape(1, -1)
        
        return audio_tensor

    def k_fold_split(self, k:int):
        folds = []
        self.fold_nums = np.array(self.fold_nums)
        for i in range(1, k+1):
            indices = np.where(self.fold_nums == i)
            folds.append(Subset(self, indices))
        return folds
