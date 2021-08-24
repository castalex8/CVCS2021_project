import ntpath
import os
from torchvision.io import read_image
from road_signs.classification import predict_class_label
import glob


# EVALUATE_FOLDER = '/home/corra/Desktop/test_images'
EVALUATE_FOLDER = '/homes/mcorradini/CVCS2021_project/road_signs/test_images'


all_images = set(os.listdir(EVALUATE_FOLDER))
crossing_images = set([ntpath.basename(f) for f in glob.glob(EVALUATE_FOLDER + '/crosswalking__*')])
total_cross_images = len(crossing_images)
total_other_images = len(all_images.difference(crossing_images))


def evaluate_mapillary_class():
    positive_label = 0
    negative_label = 0

    for positive_img in crossing_images:
        img = read_image(os.path.join(EVALUATE_FOLDER, positive_img))
        label = predict_class_label(img)
        if 'information--pedestrians-crossing' in label:
            positive_label += 1

    print(f'Score positive: {positive_label / total_cross_images * 100}%')

    for negative_img in all_images.difference(crossing_images):
        img = read_image(os.path.join(EVALUATE_FOLDER, negative_img))
        label = predict_class_label(img)
        if 'information--pedestrians-crossing' not in label:
            negative_label += 1

    print(f'Score negative: {negative_label / total_other_images * 100}%')


def evaluate_unknown_class():
    positive_label = 0
    negative_label = 0

    for positive_img in crossing_images:
        img = read_image(os.path.join(EVALUATE_FOLDER, positive_img))
        label = predict_class_label(img)
        if 'crosswalk' == label:
            positive_label += 1

    print(f'Score positive: {positive_label / total_cross_images * 100}%')

    for negative_img in all_images.difference(crossing_images):
        img = read_image(os.path.join(EVALUATE_FOLDER, negative_img))
        label = predict_class_label(img)
        if 'crosswalk' != label:
            negative_label += 1

    print(f'Score negative: {negative_label / total_other_images * 100}%')


if __name__ == '__main__':
    print(f'Testing {total_cross_images} positive images and {total_other_images} negative')

    if os.getenv('DATASET') == 'mapillary':
        evaluate_mapillary_class()
    elif os.getenv('DATASET') == 'unknown':
        evaluate_unknown_class()
    else:
        print('ERROR: set a correct dataset [mapillary|unknown]')
