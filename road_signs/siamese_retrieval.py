import torch
from collections import defaultdict
from road_signs.cnn.SiameseNet import SiameseNet
from road_signs.datasets_utils import get_dataset, get_weights, get_formatted_image, get_device, get_retrieval_images
from road_signs.loss.CostrastiveLoss import ContrastiveLoss
from road_signs.utils.Const import MARGIN


def retrieve_top_n_results(img, max_results=10):
    ds = get_dataset()
    model = SiameseNet()
    device = get_device()
    loss_fn = ContrastiveLoss(margin=MARGIN)
    model.load_state_dict(torch.load(get_weights('retrieval_siamese'), map_location=torch.device(device.type)))
    formatted_img = get_formatted_image(img)

    model.to(device)
    loss_fn.to(device)

    model.eval()
    losses = []

    for retr_img in get_retrieval_images():
        label_img = ds['get_image_from_path'](retr_img)
        img = ds['transform'](ds['dataset'].read_image(ds['get_image'](label_img)).float()).reshape([1, 3, 32, 32])
        target = torch.tensor([1])
        if device.type == 'cuda':
            test_img, img, target = formatted_img.cuda(), img.cuda(device), target.to(device)

        output = model(formatted_img, img)
        l = loss_fn(*output, torch.tensor([1]))
        if len(losses) < max_results:
            losses.append((l, label_img))
        else:
            losses = sorted(losses, key=lambda x: x[0])
            if l < losses[-1][0]:
                losses[-1] = (l, label_img)

    return sorted(losses, key=lambda x: x[0])


def retrieve_most_similar_result(img):
    return retrieve_top_n_results(img, 1)


def retrieve_most_similar_label(img):
    ds = get_dataset()
    sort_losses = retrieve_top_n_results(img, 10)
    label_occurrences = defaultdict(int)

    for res in sort_losses:
        label_occurrences[ds['get_label'](res[1])] += 1

    return max(label_occurrences, key=label_occurrences.get)
