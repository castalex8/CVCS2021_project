import torchvision
import cv2
import os
import torch
from torch import nn

MODEL_PATH = os.getenv('MODEL_PATH')
IMG_PATH = os.getenv('IMG_PATH')


class Classifier(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(Classifier, self).__init__()
        self.cls_score = nn.Linear(input_channel, num_classes)
        self.bbox_pred = nn.Linear(input_channel, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)

        scores = self.cls_score(x)
        bbox_coord = self.bbox_pred(x)
        return scores, bbox_coord


# define model then load parameters from file
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = Classifier(in_features, num_classes=2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'])

img = cv2.imread(IMG_PATH)
img = cv2.resize(img, (1000, 600))

model.eval()

img_tensor = (torchvision.transforms.ToTensor()(img))
predictions = model([img_tensor])

for prediction in predictions:
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x_min = int(box[0].item())
            y_min = int(box[1].item())
            x_max = int(box[2].item())
            y_max = int(box[3].item())
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                          (255, 0, 0), 2)
            if label.item() == 1:
                cv2.putText(img, "person " + str(round(score.item(), 3)), (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("detections", img)
    cv2.waitKey(0)
