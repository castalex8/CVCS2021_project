import os

import pandas as pd
import torchvision.io
from torch.utils.data import Dataset
from torchvision import transforms

TRANSFORMS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


CLASSES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]


class GermanTrafficSignDatasetAbs(Dataset):
    def __init__(self, train=True, trans=None):
        self.train = train
        self.base_dir = os.getenv('GERMAN_BASE_DIR')
        self.img_labels = pd.read_csv(os.path.join(self.base_dir, 'Train.csv' if self.train else 'Test.csv'))
        self.transform = trans if trans else TRANSFORMS

    def __len__(self) -> int:
        return len(self.img_labels)

    def read_image(self, item: pd.Series) -> torchvision.io.image:
        img_path = os.path.join(self.base_dir, item.Path)
        x1 = item.values[2]
        x2 = item.values[4]
        y1 = item.values[3]
        y2 = item.values[5]
        image = torchvision.io.read_image(img_path)
        img = image[:, y1:y2, x1:x2]
        return img.double()

    def __getitem__(self, index: int):
        raise NotImplementedError
