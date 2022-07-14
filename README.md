# Pedestrians, Signs, Crosswalks Detection System
Understanding the environment is crucial in many applications, especially when dealing with car driving safety. 
Our project aims to create a system that can help autonomous car drivers in the detection of pedestrians, signs, and crosswalks to avoid collisions.

Usage instructions below, but we first highly recommend to **read** the scientific-academic-paper-style document that describe the project.

## Pedestrians
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

In case of errors, contact **Alessandro Lugari ([243111@studenti.unimore.it](mailto:243111@studenti.unimore.it))**.

## Signs
This module implements the crosswalk sign detection, retrieval and classification.

### Datasets

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

### Utility functions
Inside the files `signs/road_signs/siamese_retrieval.py` and `signs/road_signs/siamese_retrieval.py`
there are the utility functions that get the top N labels given an image as parameter.

The same utility function dedicated to the classification can be found inside the file `classification.py`.

The script `signs/road_signs/get_img_road_sign.py` is it possible to find some usage examples of these functions.

In order to use these functions you need to set the following environment variables.
> DATASET=mapillary
> GERMAN_BASE_DIR=/home/corra/Desktop/gtsrb-german-traffic-sign
> MAPILLARY_BASE_DIR=/home/corra/Desktop/Mapillary
> RETRIEVAL_EMBEDDING_DIR=/home/corra/CVCS2021_project/signs/road_signs/retrieval_image_embeddings
> RETRIEVAL_IMAGES_DIR=/home/corra/CVCS2021_project/signs/road_signs/retrieval_images
> UNKNOWN_BASE_DIR=/home/corra/Desktop/Unknown
> WEIGHTS_DIR=/home/corra/CVCS2021_project/signs/road_signs/weights


## Crosswalks

This module implements the crosswalks detection. For any doubt mail to **Alessandro Castellucci** (**[228058@studenti.unimore.it](mailto:228058@studenti.unimore.it)**).

### Dataset

The script `crosswalks/code/dataset.py` is used for extracting 600 images each for 05, 06, 26, and 35 subfolders videos of the
[DR(eye)VE dataset](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8). For each 
subdataset only 480 frames are preserved for the detection task. Already extracted images with ground truth
(annotated by hand) is available [here](https://drive.google.com/file/d/1BzCKoo5XEhBw5tsFgOU-Bku2Ely7YKdi/view?usp=sharing).
The script takes three arguments:
- `--numVideo` for specify the video (or subfolder) number of the video you want to use for the images extraction
  (e.g. `--numVideo 05`).
- `--pathIn` for specify the first part of the absolute path where the original `DREYEVE_DATA` folder of the entire dataset is present, e.g. `--pathIn D:\Utenti\user\Desktop`, in this example
in the `Desktop` folder the original `DREYEVE_DATA` folder of the entire dataset is present.
- `--pathOut` for specify the output absolute path to store the images. **N.B.** So that all works fine you have to
use the project path, e.g. `--pathOut D:\Utenti\user\Desktop\CVCS2021_project\crosswalks\dataset\dreyeve\06` if you are extracting the video 06.

This procedure extracts 600 unlabeled images from the selected video. You can neglect this part if you download the already extracted 
labeled images [here](https://drive.google.com/file/d/1BzCKoo5XEhBw5tsFgOU-Bku2Ely7YKdi/view?usp=sharing). Take care to store the `dataset`
folder in the right project path, overwriting the already present one, that is only for some representative analysis.

### crosswalks/code/crosswalks_analysis.py

It performs the detection task. Can be executed in two ways:
- no argument passed, it executes the detection on only one image, specified in the program variables `FOLDER` and `FILENAME`. 
The image has to be already in the right path (as described in the above section). At the end will be printed the detection outcome and
if a crosswalk is detected an output image will be stored in the subfolder `found`. E.g.
```
FOLDER='05'
FILENAME='05_320.png'
```
- `--numFolder` argument specified. All images of the subdataset specified are analyzed, at the end evaluation results
are printed, a file containing predictions is generated and successful detections are stored always in the `found` subfolder. E.g. `--numFolder 26`.

### crosswalks/code/evaluation.py

The script prints the evaluation results analyzing the ground truth and predictions text files of the specified `--numFolder` subdataset.
Measures: True Positives, False Positives, True Negatives, False Negatives, Precision, Recall, True Positive Rate,
True Negative Rate, Accuracy, G-Mean, F-Measure. E.g. `--numFolder 26`.
