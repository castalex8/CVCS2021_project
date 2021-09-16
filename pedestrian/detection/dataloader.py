from torch.utils.data import Dataset
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import json

ANNOTATION_FILE = os.getenv("ANNOTATION_FILE")
IMAGES_FILE = os.getenv("IMAGES_FILE")


class CocoDataset(Dataset):
    def __init__(self, annotation_file, show_bbox=False):
        self.showbbox = show_bbox
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(
            self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=['person']))
        ))
        # self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_image = self.ids[index]

        # load image
        img_name = self.coco.loadImgs(id_image)[0]["file_name"]
        img = Image.open(os.path.join(IMAGES_FILE, img_name)).convert("RGB")

        # load annotations
        annotations_id = self.coco.getAnnIds(imgIds=id_image)
        annotations = self.coco.loadAnns(annotations_id)

        return img, annotations


# Test funzioni len e getitem tramite iterazione
dataset = CocoDataset(ANNOTATION_FILE, show_bbox=True)

print(f"numero di immagini: {len(dataset)}")

image, target = dataset[5]
i = 0
for image, target in dataset:
    print(f"Immagine #{i}")
    i += 1
    print(json.dumps(target[0]["bbox"], indent=4))
    print("\n")

    if dataset.showbbox:
        x_min, y_min = target[0]["bbox"][0], target[0]["bbox"][1]
        x_max, y_max = x_min + target[0]["bbox"][2], y_min + target[0]["bbox"][3]
        draw = ImageDraw.Draw(image)
        draw.rectangle(((x_min, y_min), (x_max, y_max)))

    plt.imshow(image)
    plt.show()
