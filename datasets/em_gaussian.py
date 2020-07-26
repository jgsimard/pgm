import os
import numpy as np
import requests, zipfile, io

from torch.utils.data import Dataset

class EMGaussianDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        root = "hwk3data/EMGaussian"
        filename_train = root + ".train"
        filename_test = root + ".test"
        if os.path.isfile(filename_train) and os.path.isfile(filename_test):
            train = np.loadtxt(filename_train)
            test = np.loadtxt(filename_test)
        else:
            r = requests.get("http://www.iro.umontreal.ca/~slacoste/teaching/ift6269/A18/notes/hwk3data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            train = np.loadtxt(z.open(filename_train))
            test = np.loadtxt(z.open(filename_test))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = EMGaussianDataset('/root/')
    print(len(dataset))
    print(dataset[420])


    def get_datasets_hw3():

        return train, test