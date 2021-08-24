from torch.nn import PairwiseDistance
from road_signs.cnn.TripletNet import TripletNet
from road_signs.datasets_utils import *


ds = get_dataset()
device = get_device()
loss_fn = PairwiseDistance()
model = TripletNet()
model.load_state_dict(torch.load(get_weights('retrieval_triplet'), map_location=torch.device(device.type)))

model.to(device)
loss_fn.to(device)
model.eval()


def get_embedding_from_img(img1, img2, img3):
    if device.type == 'cuda':
        img1, img2, target = img1.cuda(), img2.cuda(), img3.cuda(),

    return model(img1, img2, img3)


def get_embedding_from_img_path(img1, img2, img3):
    img1 = get_image_from_path(ds, img1)
    img2 = get_image_from_path(ds, img2)
    img3 = get_image_from_path(ds, img3)

    return get_embedding_from_img(img1, img2, img3)


def retrieve_triplet_top_n_results(img, max_results=10):
    formatted_img = get_formatted_image(img)
    retrieval_images = get_retrieval_images()
    img_embedding, _, _ = get_embedding_from_img(formatted_img, formatted_img, formatted_img)
    
    losses = []
    i = 0

    while i < len(retrieval_images):
        if i >= len(retrieval_images) - 2:
            if i == len(retrieval_images) - 1:
                retr_img = retrieval_images[i]
                retr_embedding1, _, _ = get_embedding_from_img_path(retr_img, retr_img, retr_img)
                losses = update_losses(
                    loss_fn(img_embedding, retr_embedding1), losses, max_results, ds['get_image_from_path'](retr_img)
                )
            else:
                retr_embedding1, retr_embedding2, _ = get_embedding_from_img_path(
                    retrieval_images[i], retrieval_images[i + 1], retrieval_images[i + 1]
                )
                losses = update_losses(
                    loss_fn(img_embedding, retr_embedding1), losses, max_results, ds['get_image_from_path'](retrieval_images[i])
                )
                losses = update_losses(
                    loss_fn(img_embedding, retr_embedding2), losses, max_results, ds['get_image_from_path'](retrieval_images[i + 1])
                )
        else:
            retr_embedding1, retr_embedding2, retr_embedding3 = get_embedding_from_img_path(
                retrieval_images[i], retrieval_images[i + 1], retrieval_images[i + 2]
            )
            losses = update_losses(
                loss_fn(img_embedding, retr_embedding1), losses, max_results, ds['get_image_from_path'](retrieval_images[i])
            )
            losses = update_losses(
                loss_fn(img_embedding, retr_embedding2), losses, max_results, ds['get_image_from_path'](retrieval_images[i + 1])
            )
            losses = update_losses(
                loss_fn(img_embedding, retr_embedding3), losses, max_results, ds['get_image_from_path'](retrieval_images[i + 2])
            )

        i += 3

    return sorted(losses, key=lambda x: x[0])
