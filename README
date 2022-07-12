# Pedestrians, Signs, Crosswalks Detection System
Understanding the environment is crucial in many applications, especially when dealing with car driving safety. 
Our project aims to create a system that can help autonomous car drivers in the detection of pedestrians, signs, and crosswalks to avoid collisions.


## pedestrian
This is the module that take care of the pedestrian detection task.
It is structured into a notebook and an inference file.
The notebook is designed to be executed sequentially and here are the following step in order to obtain a file with the network weights:
1. If not, copy the dataset files into the same notebook's folder, for each dataset you want to use
2. Define the dataset class
3. Instantiate the dataloaders
4. Model setup for training
    - define backbone
    - define the head of the model (we use a simple network with only one layer)
    - set the number of trainable backbone layers: the number of layers frozen during training (default set to 5 - which is also the maximum)
    - define the optizer and set the parameters
    - define (optionally) the learing rate scheduler, whose task is to simply decrease by a certain factor the learning rate value every N epochs
    - start training process
5. Save the model
6. Show model output (works only in Google Colab)

If you don't need to train a model you can use the inference file, which is used for plotting detection, same as the last point in the previous pipeline.
What you need is the test dataset (we define classes for COCO and PennFudan dataset, if you use a different dataset you must define a new dataset class)
and the model, which must be a .pth file.
You will get in output the images with the eventual bounding box that locate the pedestrians in the image.
For further information check the [PyTorch documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html) related to the 
loading procedure from a checkpoint.


## signs
This module implements the crosswalk sign detection, retrieval and classification.

### datasets

This project use the following datasets. They are required to run properly the project.
- [Mapillary Dataset](https://www.mapillary.com/dataset/trafficsign)
- [German Traffic Sign Recognition Benchmark (GTSRB) Dataset](https://benchmark.ini.rub.de/)
- [Unknown dataset](https://makeml.app/datasets/road-signs)


**N.B.: in the following example, the environment variables will refer to the location of the datasets.
The following variables have to set to the path of each dataset downloaded from the official website.**
> GERMAN_BASE_DIR=/path/to/gtsrb-german-traffic-sign
> MAPILLARY_BASE_DIR=/path/to/Desktop/Mapillary
> UNKNOWN_BASE_DIR=/hpath/to/Unknown


### CNN and Datasets implementation

The implementation of the custom neural networks could be found under the folder `signs/road_signs/cnn`.
The basic neural network is called `RoadSignNet`.
The `SiameseNet` and the `TripletNet` are build on top of the `RoadSignNet`.

The folder `signs/road_signs/` contains a folder that implements the following functions for each dataset:
- A torch Dataset, that it implements subclasses for the tasks of retrieval and classification;
- The train functions, both for the tasks of retrieval and classification.

The folder `signs/weights/` contains the weights saved after the various training of the CNNs.
The file `weights.txt` contains the configuration of the CNNs during the training.

### Create the retrieval embeddings

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

3. After creating the embedding, you can use the retrieval functions in siamese_retrieval and triplet_retrieval files.


### Training the CNNs

In order to training the CNNs with a certain dataset, go under the proper folder.
For instance, in order to train the CNNs for classification using the German Dataset, 
run the script `signs/road_signs/German/train/TrainClassification.py` 
with the following environment variable:
> GERMAN_BASE_DIR=/path/to/gtsrb-german-traffic-sign

Instead, in order to train the siamese or the triplet network run the 
`signs/road_signs/German/train/train[Siemse|Triplet].py` script, with the same environment variable.


### Test the CNNs

There are 2 customizable scripts in order to test the performance of the CNNs.
- `evaluate_classification.py`: this script shows the stats of the classification given the specific weights.
It calculates the precision, recall, accuracy and f-score.
In order to properly run the scripts, set the following environment variables:
> GERMAN_BASE_DIR=/path/to/gtsrb-german-traffic-sign
> MAPILLARY_BASE_DIR=/path/to/Desktop/Mapillary
> RETRIEVAL_IMAGES_DIR=/path/to/CVCS2021_project/road_signs/retrieval_images
> UNKNOWN_BASE_DIR=/hpath/to/Unknown

- `get_ap_map.py`: this script shows the stats of the classification given a certain dataset.
It calculates the precision, recall, accuracy and f-score.
In order to properly run the scripts, set the following environment variables:
> DATASET=mapillary
> GERMAN_BASE_DIR=/path/to/gtsrb-german-traffic-sign
> MAPILLARY_BASE_DIR=/path/to/Mapillary
> RETRIEVAL_IMAGES_DIR=/path/to/CVCS2021_project/signs/road_signs/retrieval_images
> UNKNOWN_BASE_DIR=/path/to/Unknown
> WEIGHTS_DIR=/path/to/CVCS2021_project/signs/road_signs/weights

Choose the dataset you want to test using the `DATASET` variable.
The script will create the embeddings online, and it will take minutes to finish.

For any issues, report to **Matteo Corradini ([204329@studenti.unimore.it](mailto:204329@studenti.unimore.it))**.

## crosswalks

