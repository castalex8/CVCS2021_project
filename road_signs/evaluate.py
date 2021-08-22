import os

from torchvision.io import read_image

from road_signs.classification import predict_class_label

# EVALUATE_FOLDER = '/home/corra/Desktop/test_images'
EVALUATE_FOLDER = '/homes/mcorradini/CVCS2021_project/road_signs/test_images'


crossing_images = os.listdir(EVALUATE_FOLDER)
total_images = len(crossing_images)


def evaluate_mapillary_class():
    positive_label = 0
    for test_img in crossing_images:
        img = read_image(os.path.join(EVALUATE_FOLDER, test_img))
        label = predict_class_label(img)
        if 'information--pedestrians-crossing' in label:
            positive_label += 1

    print(f'Score: {positive_label / total_images * 100}%')


def evaluate_unknown_class():
    positive_label = 0
    for test_img in crossing_images:
        img = read_image(os.path.join(EVALUATE_FOLDER, test_img))
        label = predict_class_label(img)
        if 'crosswalk' == label:
            positive_label += 1

    print(f'Score: {positive_label / total_images * 100}%')


if __name__ == '__main__':
    if os.getenv('DATASET') == 'mapillary':
        evaluate_mapillary_class()
    elif os.getenv('DATASET') == 'unknown':
        evaluate_unknown_class()
    else:
        print('ERROR: set a correct dataset [mapillary|unknown]')
