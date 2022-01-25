import json
import ntpath
import os

import torchvision.io
from torch.utils.data import Dataset
from torchvision import transforms


TRANSFORMS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


CLASSES = [
    'regulatory--no-heavy-goods-vehicles--g2', 'regulatory--one-way-right--g3', 'regulatory--stop--g1',
    'regulatory--no-left-turn--g1', 'regulatory--no-entry--g1', 'regulatory--yield--g1',
    'regulatory--no-parking--g2', 'regulatory--maximum-speed-limit-60--g1', 'complementary--keep-left--g1',
    'warning--pedestrians-crossing--g4', 'regulatory--no-stopping--g15', 'information--parking--g1',
    'regulatory--one-way-left--g1', 'complementary--chevron-right--g1', 'regulatory--priority-road--g4',
    'regulatory--no-heavy-goods-vehicles--g4', 'regulatory--maximum-speed-limit-70--g1',
    'regulatory--no-u-turn--g1', 'regulatory--keep-right--g1', 'regulatory--maximum-speed-limit-40--g1',
    'regulatory--one-way-straight--g1', 'information--pedestrians-crossing--g1', 'regulatory--no-parking--g1',
    'regulatory--roundabout--g1', 'regulatory--maximum-speed-limit-50--g1', 'warning--road-bump--g1',
    'regulatory--no-heavy-goods-vehicles--g1', 'complementary--obstacle-delineator--g2', 'warning--curve-left--g2',
    'complementary--chevron-left--g1', 'complementary--go-right--g1', 'regulatory--maximum-speed-limit-80--g1',
    'warning--other-danger--g1', 'regulatory--no-overtaking--g5', 'regulatory--maximum-speed-limit-30--g1',
    'warning--curve-right--g2', 'regulatory--no-stopping--g2', 'regulatory--go-straight--g1', 'warning--road-bump--g2',
    'regulatory--no-parking--g5', 'regulatory--keep-right--g4', 'regulatory--keep-left--g1',
    'regulatory--height-limit--g1'
]


NUM_SAMPLES = 5000


class MapillaryDatasetAbs(Dataset):
    def __init__(self, train=True, trans=None):
        self.base_dir = os.getenv('MAPILLARY_BASE_DIR_LAB') if os.getenv('USE_LAB') else os.getenv('MAPILLARY_BASE_DIR_LOCAL')
        self.transform = trans if trans else TRANSFORMS
        self.train = train
        self.train_imgs = []
        self.test_imgs = []
        self.classes = 313
        self.labels = []
        self.BASE_ANNOTATION_DIR = os.path.join(self.base_dir, 'annotations/annotations')

        if train:
            self.train_imgs = os.listdir(os.path.join(self.base_dir, 'images.train.0'))[:NUM_SAMPLES]
            for img in self.train_imgs:
                for obj in json.load(open(os.path.join(self.BASE_ANNOTATION_DIR, ntpath.basename(img).replace('.jpg', '.json'))))['objects']:
                    if obj['label'] in CLASSES:
                        self.labels.append({**obj, **{'path': os.path.join(self.base_dir, 'images.train.0', img)}})
        else:
            self.test_imgs = os.listdir(os.path.join(self.base_dir, 'images.eval'))[:NUM_SAMPLES // 5]
            for img in self.test_imgs:
                for obj in json.load(open(os.path.join(self.BASE_ANNOTATION_DIR, ntpath.basename(img).replace('.jpg', '.json'))))['objects']:
                    if obj['label'] in CLASSES:
                        self.labels.append({**obj, **{'path': os.path.join(self.base_dir, 'images.eval', img)}})

    def __len__(self) -> int:
        return len(self.labels)

    def read_image(self, img: dict) -> torchvision.io.image:
        img_path = os.path.join(self.base_dir, img['path'])
        bb = img['bbox']
        image = torchvision.io.read_image(img_path)
        return image[:, int(bb['ymin']):int(bb['ymax']), int(bb['xmin']):int(bb['xmax'])].float()

    def __getitem__(self, index: int):
        raise NotImplementedError

