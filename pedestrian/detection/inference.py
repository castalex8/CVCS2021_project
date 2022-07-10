import torch
import torch.nn as nn
import torchvision
import cv2


# class Classifier(nn.Module):
#     def __init__(self, input_channel, num_classes):
#         super(Classifier, self).__init__()
#         # Conf 1
#         self.classification = nn.Sequential(
#             nn.Linear(input_channel, 50),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.Dropout(p=0.5),
#             nn.Linear(50, 50),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(50, num_classes)
#         )
#
#         # regression layers (for bounding boxes)
#         self.regression = nn.Sequential(
#             nn.Linear(input_channel, 50),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(50, num_classes * 4)
#         )
#
#     def forward(self, x):
#         x = x.flatten(start_dim=1)
#         scores = self.classification(x)
#         bbox_coord = self.regression(x)
#         return scores, bbox_coord


MODEL_PATH = ''
IMG_PATH = ''

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = Classifier(in_features, num_classes)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes=91)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(
        MODEL_PATH))
else:
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=torch.device('cpu')))

img = cv2.imread(IMG_PATH)
img = cv2.resize(img, (800, 500))

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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    cv2.imshow("detections", img)
    cv2.waitKey(0)
