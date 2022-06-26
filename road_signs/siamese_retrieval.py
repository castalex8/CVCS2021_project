from road_signs.cnn.SiameseNet import SiameseNet
from road_signs.datasets_utils import *
from road_signs.loss.ConstrastiveLoss import ContrastiveLoss
from road_signs.utils.Const import MARGIN


ds = get_dataset()
model = SiameseNet()
device = get_device()
loss_fn = ContrastiveLoss(margin=MARGIN)
model.load_state_dict(torch.load(get_weights('retrieval_siamese'), map_location=torch.device(device.type)))

# model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# model_conv.fc = get_road_sign_fc()
# model_conv.load_state_dict(torch.load('/home/corra/CVCS2021_project/road_signs/weigths/0022.pth', map_location=torch.device('cpu')))
# model = model_conv

model.to(device)
loss_fn.to(device)
model.eval()


def get_embedding_from_img(img1: torchvision.io.image, img2: torchvision.io.image):
    target = torch.tensor([1])

    if device.type == 'cuda':
        img1, img2, target = img1.cuda(), img2.cuda(), target.to(device)

    return model(img1, img2)


def get_embedding_from_img_path(img1: torchvision.io.image, img2: torchvision.io.image):
    # img1 = get_image_from_path(ds, img1)
    # img2 = get_image_from_path(ds, img2)
    # # Unknown dataset
    # img1 = ds['transform'](transforms.ToTensor()(cv2.imread(img1)[:, :, :3])).reshape([1, 3, 32, 32])
    # img2 = ds['transform'](transforms.ToTensor()(cv2.imread(img2)[:, :, :3])).reshape([1, 3, 32, 32])

    # Mapillary dataset
    img1 = ds['transform'](torchvision.io.read_image(img1).float()).reshape([1, 3, 32, 32])
    img2 = ds['transform'](torchvision.io.read_image(img2).float()).reshape([1, 3, 32, 32])

    return get_embedding_from_img(img1, img2)


def retrieve_siamese_top_n_results(img: torchvision.io.image, max_results: int = 10) -> List[dict]:
    retrieval_images = get_retrieval_images()
    formatted_img = get_formatted_image(img)
    img_embedding, _ = get_embedding_from_img(formatted_img, formatted_img)

    losses = []
    i = 0

    while i < len(retrieval_images):
        if (i + 1) < len(retrieval_images):
            retr_embedding1, retr_embedding2 = get_embedding_from_img_path(retrieval_images[i], retrieval_images[i + 1])
            losses = update_losses(
                loss_fn(img_embedding, retr_embedding1, torch.tensor([1]).to(device)),
                # losses, max_results, ds['get_image_from_path'](retrieval_images[i])
                losses, max_results, retrieval_images[i]
            )

            losses = update_losses(
                loss_fn(img_embedding, retr_embedding2, torch.tensor([1]).to(device)),
                # losses, max_results, ds['get_image_from_path'](retrieval_images[i + 1])
                losses, max_results, retrieval_images[i + 1]
            )

        i += 2

    return sorted(losses, key=lambda x: x[0])


def retrieve_siamese_top_n_results_from_embedding(img: torchvision.io.image, max_results: int = 10) -> List[dict]:
    formatted_img = get_formatted_image(img)
    img_embedding, _ = get_embedding_from_img(formatted_img, formatted_img)

    losses = []
    files = os.listdir(get_embedding_path('siamese'))

    if len(files) % 2 == 0:
        files.append("")

    for f1, f2 in list(zip(files, files[1:]))[::2]:
        losses = update_losses(
            loss_fn(img_embedding, torch.load(os.path.join(get_embedding_path(), f1)), torch.tensor([1]).to(device)),
            losses, max_results, ds['get_image_from_path'](f1[:-2])
        )

        if f2 != '':
            losses = update_losses(
                loss_fn(img_embedding, torch.load(os.path.join(get_embedding_path(), f2)), torch.tensor([1]).to(device)),
                losses, max_results, ds['get_image_from_path'](f2[:-2])
            )

    return sorted(losses, key=lambda x: x[0])
