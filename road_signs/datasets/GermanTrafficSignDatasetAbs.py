import os
from torch.utils.data import Dataset
from torchvision import transforms as trans
import pandas as pd


BASE_DIR_LOCAL = 'gtsrb-german-traffic-sign'
BASE_DIR_LAB = '/nas/softechict-nas-3/mcorradini/gtsrb-german-traffic-sign'
t = trans.Compose([
    trans.ToTensor(),
    trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class GermanTrafficSignDatasetAbs(Dataset):
    def __init__(self, train=True, transform=None, is_local=True):
        self.base_dir = BASE_DIR_LOCAL if is_local else BASE_DIR_LAB
        self.img_labels = pd.read_csv(os.path.join(self.base_dir, 'Train.csv' if train else 'Test.csv'))
        self.transform = transform if transform else t

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        raise NotImplementedError


def get_classes(is_local=True):
    classes = pd.read_csv(os.path.join(BASE_DIR_LOCAL if is_local else BASE_DIR_LAB, 'signnames.csv')).SignName
    return [c for c in classes]
