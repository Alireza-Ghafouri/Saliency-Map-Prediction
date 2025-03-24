# Saliency Map Prediction

This project implements a convolutional neural network for saliency map prediction, inspired by the **Deep ConvNet** model proposed in the paper ["Shallow and Deep Convolutional Networks for Saliency Prediction"](https://openaccess.thecvf.com/content_cvpr_2016/papers/Pan_Shallow_and_Deep_CVPR_2016_paper.pdf). The goal is to predict regions of an image that attract the most human attention.

## Features
- Implements the **Deep ConvNet** model from the referenced paper with reduced network depth to optimize memory and computation.
- Uses the **2000cat dataset**, a smaller version of the **SALICON dataset**, for training and evaluation.
- Supports flexible configuration of hyperparameters, dataset paths, and saving directories via a config file.
- Uses **L2 loss function** for training, following the methodology in the paper.
- Training and evaluation follow the **85%-15% data split**, with uniform sampling from all dataset classes.
- Outputs **loss and visualization plots** in the specified directory.

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, simply run:
```bash
python main.py
```

## Configuration
All hyperparameters, dataset paths, and saving directories can be modified in the `config` file.

## Output
- The trained model and logs will be saved in the specified path.
- Visualization results will be available after training.

## Reference
- "Shallow and Deep Convolutional Networks for Saliency Prediction" - CVPR 2016 ([Paper Link](https://openaccess.thecvf.com/content_cvpr_2016/papers/Pan_Shallow_and_Deep_CVPR_2016_paper.pdf)).
