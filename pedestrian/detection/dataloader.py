from torch.utils.data import Dataset
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import json
import torch

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
        img = Image.open(os.path.join(IMAGES_FILE, img_name))

        # load annotations
        annotations_id = self.coco.getAnnIds(imgIds=id_image)
        annotations = self.coco.loadAnns(annotations_id)

        num_objs = len(annotations)
        # build information about bounding box & area
        boxes = []
        areas = []
        labels = torch.ones((num_objs,), dtype=torch.int64)
        for i in range(num_objs):
            x_min = annotations[i]['bbox'][0]
            y_min = annotations[i]['bbox'][1]
            x_max = x_min + annotations[i]['bbox'][2]
            y_max = y_min + annotations[i]['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(annotations[i]['area'])
            labels[i] = annotations[i]['category_id']

        # transfer information to Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([id_image])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation in dict form
        Annotations = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd
        }
        return img, Annotations, id_image


# Test funzioni len e getitem tramite iterazione
dataset = CocoDataset(ANNOTATION_FILE, show_bbox=True)

print(f"numero di immagini: {len(dataset)}")

# image, target = dataset[5]
i = 0
for image, target, id_img in dataset:
    print(f"Immagine #{i}")
    i += 1
    # print(target)
    # print(target["bbox"])
    print("\n")

    if dataset.showbbox:
        for j in range(len(target['boxes'])):
            x_min, y_min = target["boxes"][j][0], target["boxes"][j][1]
            x_max, y_max = target["boxes"][j][2], target["boxes"][j][3]
            if target['labels'][j] == 1:
                draw = ImageDraw.Draw(image)
                draw.rectangle(((x_min, y_min), (x_max, y_max)))

    plt.imshow(image)
    plt.show()
