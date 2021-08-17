# from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from base import BaseDataLoader
from .utils import collate_fn
from .DeepLesion import DeepLesionDataset
#from sampler import SubsetSequentialSampler

import albumentations as A
import numpy as np

class DeepLesion(nn.Module):
    def __init__(self, root, validation_split=0.2, num_workers=2, 
                 dataset_type='non-specified', lesion_type='lung'):
        self.transforms = A.Compose([
                            A.ShiftScaleRotate(p=0.5),
                            A.Transpose(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
        )

        self.dataset = DeepLesionDataset(root, self.transforms, dataset_type, lesion_type)
        self.dataset_noaug = DeepLesionDataset(root, None, dataset_type, lesion_type) 
        self.n_samples = len(self.dataset)
        self.num_workers = num_workers

        self.validation_split = validation_split
        self.train_idx, self.val_idx, self.test_idx = self._split_train_val_test(self.validation_split)

    def _split_train_val_test(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        #np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_train = self.sample_num - split
        else:
            len_train = int(self.n_samples * (1-split))
            len_val = int(self.n_samples * split / 2)

        train_idx = idx_full[:len_train]
        val_idx = idx_full[len_train:len_train+len_val]
        test_idx = idx_full[len_train+len_val:]

        return train_idx, val_idx, test_idx

    def get_val_test_dataloaders(self):
        val_dataset = torch.utils.data.Subset(self.dataset_noaug, self.val_idx)
        test_dataset = torch.utils.data.Subset(self.dataset_noaug, self.test_idx)

        data_loader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
        data_loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

        return data_loader_val, data_loader_test

    def get_train_dataset(self):
        return torch.utils.data.Subset(self.dataset, self.train_idx)