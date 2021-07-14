import torch
import torchvision
from torchvision import transforms, utils
import numpy as np
import os
import random
import glob

import torch
from torch.utils.data import Dataset


class LogoDataset(Dataset):
    def __init__(self,root):
        self.root = root
        if self.root is not None:
            self.data_dir = os.path.join(self.root)
        subsets = os.listdir(self.data_dir)
        self.file_names = []
        for i in subsets:
            file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.data_dir, i, "*.*"))]
            file_names.sort()
            self.file_names += [os.path.join(i, f) for f in file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        pass