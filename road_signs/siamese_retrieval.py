from road_signs.cnn.SiameseNet import SiameseNet
from road_signs.datasets_utils import *
from road_signs.loss.CostrastiveLoss import ContrastiveLoss
from road_signs.utils.Const import MARGIN


ds = get_dataset()
model = SiameseNet()
device = get_device()
loss_fn = ContrastiveLoss(margin=MARGIN)
model.load_state_dict(torch.load(get_weights('retrieval_siamese'), map_location=torch.device(device.type)))
model.to(device)
loss_fn.to(device)


def get_embedding_from_img(img1, img2):
    target = torch.tensor([1])

    if device.type == 'cuda':
        img1, img2, target = img1.cuda(), img2.cuda(), target.to(device)

    return model(img1, img2)


def get_embedding_from_img_path(img1, img2):
    img1 = get_image_from_path(ds, img1)
    img2 = get_image_from_path(ds, img2)

    return get_embedding_from_img(img1, img2)


def retrieve_siamese_top_n_results(img, max_results=10):
    retrieval_images = get_retrieval_images()
    formatted_img = get_formatted_image(img)
    img_embedding, _ = get_embedding_from_img(formatted_img, formatted_img)
    model.eval()
    losses = []
    i = 0

    while i < len(retrieval_images):
        retr_embedding1, retr_embedding2 = get_embedding_from_img_path(retrieval_images[i], retrieval_images[i + 1])
        losses = update_losses(
            loss_fn(img_embedding, retr_embedding1, torch.tensor([1])),
            losses, max_results, ds['get_image_from_path'](retrieval_images[i])
        )
        losses = update_losses(
            loss_fn(img_embedding, retr_embedding2, torch.tensor([1])),
            losses, max_results, ds['get_image_from_path'](retrieval_images[i + 1])
        )
        i += 2

    return sorted(losses, key=lambda x: x[0])
