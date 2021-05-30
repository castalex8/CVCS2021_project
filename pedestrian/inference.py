import torch
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os


WEIGHT_PATH = os.getenv("WEIGHT_PATH")
ANNOTATION_FILE = os.getenv("ANNOTATION_FILE")
FONT_PATH = os.getenv("FONT_PATH")
SCORE_THRESHOLD = 0.2
FONT_SIZE = 20
FOCAL_LENGTH = 35
OBJECT_HEIGHT = 1700


def get_model(num_classes):
    pretrained = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = pretrained.roi_heads.box_predictor.cls_score.in_features
    pretrained.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return pretrained


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
coco = COCO(ANNOTATION_FILE)
catIds = coco.getCatIds(catNms=['person', 'traffic light'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
I = io.imread(img['coco_url'])

preprocess = torchvision.transforms.ToTensor()

Ipil = Image.fromarray(I)
draw = ImageDraw.Draw(Ipil)

model = get_model(2)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    prediction = model([(preprocess(I)).to(device)])

print(prediction)
boxes = prediction[0]['boxes']
scores = prediction[0]['scores']

plt.axis('off')
plt.imshow(Ipil)
plt.show()

for i, box in enumerate(boxes):
    print(box)
    if scores[i] > SCORE_THRESHOLD:
        fnt = ImageFont.truetype(font=FONT_PATH, size=FONT_SIZE)
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=5)
        distance = (FOCAL_LENGTH * OBJECT_HEIGHT) / (box[3] - box[1])
        draw.text((box[0], box[1]), f"distance: {( distance / 10)}", font=fnt)
plt.axis('off')
plt.imshow(Ipil)
plt.show()
