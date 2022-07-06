### HOW TO USE LIBRARIES

In order to properly use the retrieval and classifications libraries
you need to download the datasets and set the correct environment variables.

Inside the mapillary folder must exist the following folders:
- annotation
- images.train.0 (first train dataset)
- images.eval (evaluation dataset)


The variables are the following:
- GERMAN_BASE_DIR (es. GERMAN_BASE_DIR=/nas/softechict-nas-3/user/gtsrb-german-traffic-sign) 
- UNKNOWN_BASE_DIR
- MAPILLARY_BASE_DIR
- DATASET: set the dataset you want to use. The values acceptable values are:
  - mapillary
  - unknown
  - german


Resources:
- German Dataset: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download
- Unknown Dataset: https://www.kaggle.com/andrewmvd/road-sign-detection/download
- Mapillary Dataset: https://www.mapillary.com/dataset/trafficsign
