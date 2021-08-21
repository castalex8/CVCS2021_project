import torch
from road_signs.cnn.RoadSignNet import RoadSignNet
from road_signs.datasets_utils import get_dataset, get_weights, get_predicted_class, get_formatted_image


def predict_class(img):
    ds = get_dataset()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RoadSignNet(classes=len(ds['classes']))
    model.load_state_dict(torch.load(get_weights('classification'), map_location=torch.device(device.type)))
    model.to(device)
    model.eval()

    outputs = model(get_formatted_image(img))
    _, predictions = torch.max(outputs, 1)

    return predictions


def predict_class_label(img):
    return get_predicted_class(int(predict_class(img)))
