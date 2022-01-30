from collections import defaultdict
from typing import List, Tuple

import numpy as np


def create_test_pairs(labels: List[dict], img_classes: defaultdict) -> List[List[dict, dict]]:
    positive_pairs = [[] for _ in range(len(labels) // 2)]
    negative_pairs = [[] for _ in range(len(labels) - len(positive_pairs))]
    for i in range(len(positive_pairs)):
        # Match
        pos_img = labels[i]
        positive_pairs[i].append(pos_img)
        pos_img2 = pos_img
        while pos_img2['path'] == pos_img['path']:
            pos_img2 = img_classes[pos_img['label']][np.random.randint(0, len(img_classes[pos_img['label']]))]

        positive_pairs[i].append(pos_img2)
        positive_pairs[i].append(1)

    for i in range(len(negative_pairs)):
        # No match
        pos_img = labels[i]
        negative_pairs[i].append(pos_img)
        negative_label = pos_img['label']
        negative = None
        while negative_label == pos_img['label']:
            negative_label = list(img_classes.keys())[np.random.randint(0, len(img_classes))]
            negative = img_classes[negative_label][np.random.randint(0, len(img_classes[negative_label]))]

        negative_pairs[i].append(negative)
        negative_pairs[i].append(0)

    return positive_pairs + negative_pairs


def create_online_training_couple(img1, img_classes) -> Tuple[dict, dict, int]:
    target = np.random.randint(0, 2)
    img2 = None
    if target == 1:
        # Match
        img2 = img1
        while img2['path'] == img1['path']:
            img2 = img_classes[img1['label']][np.random.randint(0, len(img_classes[img1['label']]))]
    else:
        # No match
        label2 = img1['label']
        while label2 == img1['label']:
            label2 = list(img_classes.keys())[np.random.randint(0, len(img_classes.keys()))]
            img2 = img_classes[label2][np.random.randint(0, len(img_classes[label2]))]
    
    return img2, img2, target


def create_test_triplets(labels: List[dict], img_classes: defaultdict) -> List[List[dict, dict, dict]]:
    triplets = [[] for _ in range(len(labels))]
    for i in range(len(labels)):
        anchor = labels[i]
        triplets[i].append(anchor)

        positive = anchor
        while positive['path'] == anchor['path']:
            positive = img_classes[anchor['label']][np.random.randint(0, len(img_classes[anchor['label']]))]
        triplets[i].append(positive)

        negative = anchor
        while negative['label'] == anchor['label']:
            negative_label = list(img_classes.keys())[np.random.randint(0, len(img_classes))]
            negative = img_classes[negative_label][np.random.randint(0, len(img_classes[negative_label]))]

        triplets[i].append(negative)

    return triplets


def create_online_training_triplet(anchor, img_classes) -> Tuple[dict, dict, dict]:
    positive = anchor

    while positive['path'] == anchor['path']:
        positive = img_classes[anchor['label']][np.random.randint(0, len(img_classes[anchor['label']]))]

    negative = anchor
    while negative['label'] == anchor['label']:
        negative_label = list(img_classes.keys())[np.random.randint(0, len(img_classes))]
        negative = img_classes[negative_label][np.random.randint(0, len(img_classes[negative_label]))]

    return anchor, positive, negative
