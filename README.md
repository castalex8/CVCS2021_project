# Pedestrians, Signs, Crosswalks Detection System
Understanding the environment is crucial in many applications, especially when dealing with car driving safety. 
Our project aims to create a system that can help autonomous car drivers in the detection of pedestrians, signs, and crosswalks to avoid collisions.

## pedestrian

## signs
This module implements the crosswalk sign detection, retrieval and classification.

This project use the following datasets. They are required to run properly the project.
- [Mapillary Dataset](https://www.mapillary.com/dataset/trafficsign)
- [German Traffic Sign Recognition Benchmark (GTSRB) Dataset](https://benchmark.ini.rub.de/)
- [Unknown dataset](https://makeml.app/datasets/road-signs)


In order to create the retrieval embeddings, follow these steps.

1. Create the retrieval set.
Export the following environment variables before running `create_retrieval_set.py`
> GERMAN_BASE_DIR=/path/to/gtsrb-german-traffic-sign
> MAPILLARY_BASE_DIR=/path/to/Desktop/Mapillary
> RETRIEVAL_IMAGES_DIR=/path/to/CVCS2021_project/road_signs/retrieval_images
> UNKNOWN_BASE_DIR=/hpath/to/Unknown

This script will create all the retrieval road signs images in the folder `retrieval_images`,
divided by the proper dataset.

2. Create the embedding set for each dataset.
Add to the previous environment variables the following ones:
> RETRIEVAL_EMBEDDING_DIR=/path/to/CVCS2021_project/signs/road_signs/retrieval_image_embeddings
> WEIGHTS_DIR=/path/to/CVCS2021_project/signs/road_signs/weights

Then run the scripts `create_{dataset}_retrieval_embedding_set.py` to create the embeddings for each dataset.

Before running the script, set the environment variable DATASET for each script. 
For instance, in order to run `create_{german}_retrieval_embedding_set.py`, export
> DATASET=german

The scripts will create the embedding in the folder `retrieval_image_embeddings` for the siamese and triplet network
in the proper folder, divider by the proper dataset.

3. 

For any issues, report to **Matteo Corradini ([204329@studenti.unimore.it](mailto:204329@studenti.unimore.it))**.

## crosswalks

