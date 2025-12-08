# Tiny ImageNet CNN Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A Convolutional Neural Network (CNN) for classifying images from the **Tiny ImageNet** dataset.  
This model achieves **79% test accuracy** without self-attention or hyperparameter optimization. All code is implemented in **Jupyter Notebook** for easy experimentation.

---

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Model](#model)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- CNN implemented from scratch using PyTorch  
- Trains on Tiny ImageNet dataset  
- Modular Jupyter Notebook for step-by-step execution  
- Achieves 79% test accuracy without self-attention or hyperparameter tuning  
- Easy to extend for experiments with attention modules or hyperparameter optimization  

---

## Dataset

- **Tiny ImageNet** ([website](https://tiny-imagenet.herokuapp.com/))  
- 200 classes, 500 training images per class  
- 64x64 RGB images  

> **Note:** The dataset is not included in this repository. Download it separately and place it in the designated `data/` folder.

---

## Model

- Standard CNN architecture with multiple convolutional layers, ReLU activations, and max pooling  
- Fully connected layers at the end for classification  
- No self-attention or advanced hyperparameter optimization  

---

## Requirements

- Python 3.x  
- PyTorch 2.x  
- torchvision  
- numpy  
- matplotlib  
- pandas  
- Jupyter Notebook  

You can install all Python dependencies with:

```bash
pip install -r requirements.txt
