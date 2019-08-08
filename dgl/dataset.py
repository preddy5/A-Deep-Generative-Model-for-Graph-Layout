
import glob
from torch.utils.data import Dataset

def get_transform():
    transforms_list = []
    transforms_list += [transforms.ToTensor(), ]
    return transforms.Compose(transforms_list)

def normalize(pos):
    return pos/np.max(pos)

class GraphData(Dataset):
    def __init__(self, name, folder):
        self.files = glob.glob(folder+ name + '*')  # [:1000]
        self.data = []
        self.adj = np.load(name+'_adj.npy')
        for i in self.files:
            self.data.append(np.load(i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        pos = normalize(pos)
        transform_augment = get_transform()
        return transform_augment(pos), transform_augment(self.adj)

