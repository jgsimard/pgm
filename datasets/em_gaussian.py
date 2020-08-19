import os
import numpy as np
from torch.utils.data import Dataset
import torch


class EMGaussianDataset(Dataset):
    def __init__(self, data_root, train=True):
        file = "train.txt" if train else "test.txt"
        root = os.path.join(data_root, file)
        if os.path.isfile(root):
            data = np.loadtxt(root)
        else:
            raise FileNotFoundError(f"root not fount at : {data_root}")
        self.samples = torch.from_numpy(data)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.samples


if __name__ == '__main__':
    dataset = EMGaussianDataset('data/EMGaussian')
    print(len(dataset))
    print(dataset[0])
