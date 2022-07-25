import os.path

import torchvision.io

from signs.road_signs.classification import predict_class_label
from signs.road_signs.siamese_retrieval import retrieve_siamese_top_n_results_from_embedding
from signs.road_signs.triplet_retrieval import retrieve_triplet_top_n_results_from_embedding
import argparse


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--file', '-f', help='Insert the path of the image to predict the class', required=True)
    args = a.parse_args()

    if not os.path.exists(args.file):
        print('File not found')
        return

    img = torchvision.io.read_image(args.file)

    print('Siamese results:', end=' ')
    print(', '.join([res[1] for res in retrieve_siamese_top_n_results_from_embedding(img)]))

    print('Triplet results:', end=' ')
    print(', '.join([res[1] for res in retrieve_triplet_top_n_results_from_embedding(img)]))

    print('Predicted class:', predict_class_label(img))


if __name__ == '__main__':
    main()
