import os
import numpy as np
from skimage import io, transform, exposure
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


BASE_DIR_LOCAL = 'gtsrb-german-traffic-sign'
BASE_DIR_LAB = '/nas/softechict-nas-3/mcorradini/gtsrb-german-traffic-sign'
t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class GermanTrafficSignDatasetAbs(Dataset):
    def __init__(self, train=True, trans=None, is_local=True):
        self.train = train
        self.base_dir = BASE_DIR_LOCAL if is_local else BASE_DIR_LAB
        self.img_labels = pd.read_csv(os.path.join(self.base_dir, 'Train.csv' if self.train else 'Test.csv'))
        self.transform = trans if trans else t

    def __len__(self):
        return len(self.img_labels)

    def format_image(self, item_path):
        img_path = os.path.join(self.base_dir, item_path)
        image = io.imread(img_path)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
        image.astype(np.double)
        return image

    def __getitem__(self, index):
        raise NotImplementedError


def get_classes(is_local=True):
    classes = pd.read_csv(os.path.join(BASE_DIR_LOCAL if is_local else BASE_DIR_LAB, 'signnames.csv')).SignName
    return [c for c in classes]
