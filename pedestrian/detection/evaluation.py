from dataloader import CocoDataset
import os
import torch
import torchvision
from pedestrian import inference
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ANNOTATION_FILE = os.getenv("ANNOTATION_FILE")
WEIGHT_PATH = os.getenv("WEIGHT_PATH")
RESULT_FILENAME = os.getenv("RESULT_FILENAME")
SCORE_THRESHOLD = 0

dataset = CocoDataset(ANNOTATION_FILE)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

preprocess = torchvision.transforms.ToTensor()

print("### Loading model ###")
model = inference.get_model(2)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))
model.eval()


with open(RESULT_FILENAME, "w") as result_file:
    result_file.write("[")

i = 0
MAX_IMG = 10
last_pos = 0
images_id = list()
for image, annotation, image_id, category_id in dataset:
    if i == MAX_IMG:
        break
    images_id.append(image_id)
    with torch.no_grad():
        prediction = model([(preprocess(image)).to(device)])

    boxes = prediction[0]["boxes"].tolist()
    scores = prediction[0]["scores"].tolist()

    with open(RESULT_FILENAME, "a") as result_file:
        for box, score in zip(boxes, scores):
            if score > SCORE_THRESHOLD:
                dict_info = {
                    "image_id": image_id,
                    "category_id": category_id[0],
                    "bbox": box,
                    "score": score
                }
                print(json.dumps(dict_info))
                result_file.write(json.dumps(dict_info))
                result_file.write(",")
                '''
                print(prediction, "\n\n\n", prediction[0], "\n\n\n")
                x_min, y_min = prediction[0]["boxes"][0], prediction[0]["boxes"][1]
                x_max, y_max = prediction[0]["boxes"][2] - x_min, prediction[0]["boxes"][3] - y_min
                print(x_min, x_max, y_min, y_max)
                draw = ImageDraw.Draw(image)
                draw.rectangle(((x_min, y_min), (x_max, y_max)))
            plt.imshow(image)
            plt.show()'''

    i += 1

with open(RESULT_FILENAME, "rb+") as result_file:
    result_file.seek(-1, os.SEEK_END)
    result_file.truncate()

with open(RESULT_FILENAME, "a") as result_file:
    result_file.write("]")

# evaluation of performance using CocoEval
cocoGt = COCO(ANNOTATION_FILE)
cocoDt = cocoGt.loadRes(RESULT_FILENAME)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = images_id
cocoEval.params.catIds = [1]  # cocoGt.getCatIds(catNms=['person', 'stop sign'])
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
