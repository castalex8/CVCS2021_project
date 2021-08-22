import os
from collections import defaultdict

import torch
from torch.nn import TripletMarginWithDistanceLoss, PairwiseDistance
from road_signs.cnn.TripletNet import TripletNet
from road_signs.datasets_utils import get_dataset, get_formatted_test_image, get_device, get_retrieval_images, \
    get_weights, get_formatted_image
from road_signs.utils.Const import MARGIN


def retrieve_triplet_top_n_results(img, max_results=10):
    ds = get_dataset()
    device = get_device()
    loss_fn = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(), margin=MARGIN)
    model = TripletNet()
    model.load_state_dict(torch.load(get_weights('retrieval_triplet'), map_location=torch.device(device.type)))
    test_img = get_formatted_test_image()
    formatted_img = get_formatted_image(img)

    model.to(device)
    loss_fn.to(device)

    model.eval()
    losses = []

    for retr_img in get_retrieval_images():
        label_img = ds['get_image_from_path'](retr_img)
        img = ds['transform'](ds['dataset'].read_image(ds['get_image'](label_img)).float()).reshape([1, 3, 32, 32])
        output = model(test_img, img, img)
        l = loss_fn(*output, torch.tensor([1]))
        if len(losses) < 10:
            losses.append((l, label_img))
        else:
            losses = sorted(losses, key=lambda x: x[0])
            if l < losses[-1][0]:
                losses[-1] = (l, label_img)

