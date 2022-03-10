import os
import torch
import torchvision
from pedestrian import inference
import cv2
import imutils


def detect_pedestrians(imagepath, conf='cnn', enable=False):
    if enable is not True:
        return

    if conf != 'hogsvm':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # choose model based on configuration
        if conf == 'cnn1':
            WEIGHT_PATH = os.getenv("WEIGHT_PATH_CNN1")
        if conf == 'cnn2':
            WEIGHT_PATH = os.getenv("WEIGHT_PATH_CNN2")
        else:
            WEIGHT_PATH = os.getenv("WEIGHT_PATH_CNN3")
        model = inference.get_model(2)
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))
        model.eval()

        preprocess = torchvision.transforms.ToTensor()
        with torch.no_grad():
            prediction = model([(preprocess(imagepath)).to(device)])

        return prediction
    else:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        image = cv2.imread(imagepath)
        image = imutils.resize(image,
                               width=min(400, image.shape[1]))
        (regions, _) = hog.detectMultiScale(image,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)


def detect_roadsign(imagepath='', conf='', enable=False):
    return True


def detect_crossroad(imagepath='', enable=False):
    return 0


# pedestrian detection
pedestrian_bboxes = detect_pedestrians(imagepath='', conf='', enable=False)

# road sign detection
roadsign_flag = detect_roadsign(imagepath='', conf='', enable=False)

# cross road detection
crossroad = detect_crossroad(imagepath='', enable=False)

if roadsign_flag:
    for pedestrian in pedestrian_bboxes:
        # check coordinates between pedetrian and crossroad (if any)
        pass
