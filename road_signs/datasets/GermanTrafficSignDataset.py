import os
from skimage import io, exposure, transform
from torch.utils.data import Dataset
from torchvision import transforms as trans
import pandas as pd
import numpy as np


t = trans.Compose([
    trans.ToTensor(),
    # trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


BASE_DIR = 'gtsrb-german-traffic-sign'


class GermanTrafficSignDataset(Dataset):
    def __init__(self, train=True, transform=t):
        self.base_dir = BASE_DIR
        self.img_labels = pd.read_csv(os.path.join(self.base_dir, 'Train.csv' if train else 'Test.csv'))# [:1000]
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        item = self.img_labels.iloc[index]
        img_path = os.path.join(self.base_dir, item.Path)
        image = io.imread(img_path)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
        image.astype(np.double)
        label = item.ClassId

        return self.transform(image), label


def get_classes():
    classes = pd.read_csv(os.path.join(BASE_DIR, 'signnames.csv')).SignName
    return [c for c in classes]
