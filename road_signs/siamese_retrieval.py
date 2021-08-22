import torch
from road_signs.cnn.SiameseNet import SiameseNet
from road_signs.datasets_utils import get_dataset, get_weights, get_formatted_image, get_device, get_retrieval_images
from road_signs.loss.CostrastiveLoss import ContrastiveLoss
from road_signs.utils.Const import MARGIN


def get_image_from_path(ds, img):
    return ds['transform'](
        ds['dataset'].read_image(ds['get_image'](ds['get_image_from_path'](img))).float()
    ).reshape([1, 3, 32, 32])


def get_embedding_from_img(model, device, img1, img2):
    target = torch.tensor([1])

    if device.type == 'cuda':
        img1, img2, target = img1.cuda(), img2.cuda(device), target.to(device)

    return model(img1, img2)


def get_embedding_from_img_path(ds, model, device, img1, img2):
    img1 = get_image_from_path(ds, img1)
    img2 = get_image_from_path(ds, img2)

    return get_embedding_from_img(model, device, img1, img2)


def update_losses(loss_fn, losses, max_results, img_embedding, retr_embedding, label_retr_img):
    l = loss_fn(img_embedding, retr_embedding, torch.tensor([1]))
    if len(losses) < max_results:
        losses.append((l, label_retr_img))
    else:
        losses = sorted(losses, key=lambda x: x[0])
        if l < losses[-1][0]:
            losses[-1] = (l, label_retr_img)

    return losses


def retrieve_siamese_top_n_results(img, max_results=10):
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
    retrieval_images = get_retrieval_images()
    img_embedding, _ = get_embedding_from_img(model, device, formatted_img, formatted_img)
    i = 0

    while i < len(retrieval_images):
        retr_embedding1, retr_embedding2 = get_embedding_from_img_path(
            ds, model, device, retrieval_images[i], retrieval_images[i + 1]
        )
        losses = update_losses(
            loss_fn, losses, max_results, img_embedding, retr_embedding1, ds['get_image_from_path'](retrieval_images[i])
        )
        losses = update_losses(
            loss_fn, losses, max_results, img_embedding, retr_embedding2, ds['get_image_from_path'](retrieval_images[i + 1])
        )
        i += 2

    return sorted(losses, key=lambda x: x[0])
