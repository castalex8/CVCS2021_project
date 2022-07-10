import torch
import torchvision
from torch import Tensor

from signs.road_signs.cnn.RoadSignNet import RoadSignNet
from signs.road_signs.datasets_utils import get_dataset, get_weights, get_predicted_class, get_formatted_image, get_device


ds = get_dataset()
device = get_device()
model = RoadSignNet(classes=len(ds['classes']))
model.load_state_dict(torch.load(get_weights('classification'), map_location=torch.device(device.type)))
model.to(device)
model.eval()


def predict_class(img: torchvision.io.image) -> Tensor:
    outputs = model(get_formatted_image(img))
    _, predictions = torch.max(outputs, 1)

    return predictions


def predict_class_label(img: torchvision.io.image) -> str:
    return get_predicted_class(int(predict_class(img)))
