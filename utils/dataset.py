
from scipy.io import loadmat
from torch.utils.data import Dataset
import os
import os.path
import torch
import numpy as np

class diy_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        data = loadmat(data_path)
        label = loadmat(label_path)
        image = torch.from_numpy(data['matrix'])
        gt = torch.from_numpy(label['gt'])

        self.image = image
        self.gt = gt

        self.length = len(self.image)

    def __getitem__(self, index):
        return self.image[index, :, :].unsqueeze(0).float(), self.gt[index].squeeze().long()

    def __len__(self):
        return self.length
