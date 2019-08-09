
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transform():
    transforms_list = []
    transforms_list += [transforms.ToTensor(), ]
    return transforms.Compose(transforms_list)

class GraphData(Dataset):
    def __init__(self, name, folder, sample=False):
        self.files = glob.glob(folder+ name + '_data*')  # [:1000]
        self.data = []
        self.adj = np.load(folder + name+'_adj.npy', allow_pickle=True)
        for i in self.files:
            self.data.append(np.load(i, allow_pickle=True))
            print(i, np.load(i, allow_pickle=True).shape)
            if sample:
                break
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        return torch.tensor(pos).type(torch.FloatTensor), torch.tensor(self.adj).type(torch.FloatTensor)

