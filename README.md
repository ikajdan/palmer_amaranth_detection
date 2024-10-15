# Palmer Amaranth Detection Using YOLOv11

This repository contains the code for detecting Palmer Amaranth in images using YOLOv11. The code is written in Python and uses the Ultralytics library for training and testing the model.

# Dataset

The [dataset](https://weed-ai.sydney.edu.au/datasets/5c78d067-8750-4803-9cbe-57df8fae55e4) is fetched from Weed-AI. It includes imagery of crops with weeds annotated, and is available in an MS-COCO derived format with standardized agricultural metadata.

To convert the dataset to YOLO format, the `coco2yolo.py` script was used.

The dataset has been re-uploaded to Roboflow for preprocessing and augmentation.

# Training the Model

Weights and Biases is used for tracking the training process.
