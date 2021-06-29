import os
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torchvision.io import read_image


BASE_DIR_LOCAL = 'gtsrb-german-traffic-sign'
BASE_DIR_LAB = '/nas/softechict-nas-3/mcorradini/gtsrb-german-traffic-sign'
t = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def is_local():
    return os.getenv('IS_LOCAL')


class GermanTrafficSignDatasetAbs(Dataset):
    def __init__(self, train=True, trans=None):
        self.train = train
        self.base_dir = BASE_DIR_LOCAL if is_local() else BASE_DIR_LAB
        self.img_labels = pd.read_csv(os.path.join(self.base_dir, 'Train.csv' if self.train else 'Test.csv'))
        self.transform = trans if trans else t

    def __len__(self):
        return len(self.img_labels)

    def read_image(self, item_path):
        img_path = os.path.join(self.base_dir, item_path)
        image = read_image(img_path)
        return image.double()

    def __getitem__(self, index):
        raise NotImplementedError


def get_classes():
    classes = pd.read_csv(os.path.join(BASE_DIR_LOCAL if is_local() else BASE_DIR_LAB, 'signnames.csv')).SignName
    return [c for c in classes]
