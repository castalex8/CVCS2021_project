import torch
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    pretrained = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = pretrained.roi_heads.box_predictor.cls_score.in_features
    pretrained.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataDir = 'D:\\coco'
dataType = 'val2017'
annFile = '{}\\annotations\\instances_{}.json'.format(dataDir, dataType)


coco = COCO(annFile)

catIds = coco.getCatIds(catNms=['person', 'traffic light'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
I = io.imread(img['coco_url'])

preprocess = torchvision.transforms.ToTensor()

Ipil = Image.fromarray(I)
draw = ImageDraw.Draw(Ipil)

model = get_model(2)
model.load_state_dict(torch.load('D:\\weights.pth', map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    prediction = model([(preprocess(I)).to(device)])

print(prediction)
boxes = prediction[0]['boxes']

plt.axis('off')
plt.imshow(Ipil)
plt.show()

for box in boxes:
    print(box)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=5)

plt.axis('off')
plt.imshow(Ipil)
plt.show()
