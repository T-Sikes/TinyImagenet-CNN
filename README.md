# Tiny ImageNet CNN Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **Convolutional Neural Network (CNN)** for classifying images from the **Tiny ImageNet** dataset.  
This model achieves **79% test accuracy** without self-attention or hyperparameter optimization.  
All code is implemented in **Jupyter Notebook** for easy experimentation and modular execution.

---

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Model](#model)  
- [Requirements](#requirements)  
- [Installation](#installation)  

---

## Features

- CNN implemented from scratch using PyTorch  
- Trains on Tiny ImageNet dataset (15 classes included in repo)  
- Modular Jupyter Notebook for step-by-step execution  
- Achieves 79% test accuracy without self-attention or hyperparameter tuning  
- Easily extendable for attention modules or hyperparameter optimization  

---

## Dataset

- **Tiny ImageNet** ([website](https://tiny-imagenet.herokuapp.com/))  
- 15 classes, 500 training images per class (subset included)  
- 64x64 RGB images  

> **Note:** The full Tiny ImageNet dataset contains 200 classes and is not included in this repository.  
> Download the full dataset separately and place it in the `data/` folder if needed.

---

## Model

- Standard CNN architecture with multiple convolutional layers, ReLU activations, and max pooling  
- Stepped dropout with identity pooling in coupled convolutional layers  
- Fully connected layers at the end for classification  
- No self-attention or advanced hyperparameter optimization  

---

## Requirements

- **Python:** 3.11 or later  
- **PyTorch:** 2.x (CPU or GPU version; see installation notes below)  
- **torchvision**  
- **numpy**  
- **matplotlib**  
- **pandas**  
- **Jupyter Notebook**  

---

## Installation

It is recommended to use a virtual environment: (Remember to select the venv as intepreter if using VSCode)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```
Important Note

PyTorch GPU versions are platform-specific. For example, torch==2.9.1+cu130 may not be available for all CUDA versions or Python 3.11.

If a specific GPU version is not found, you can install a CPU-only version:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Though be advised this project was made with GPU compatibility in mind, no testing or compabatibility resolution has been done for CPU-only implementation.

Due to this, it is HIGHLY RECCOMENDED to manually install the compatible version of torch for your hardware before installing dependencies.

You can find compatible versions for NVIDIA GPU's [Here](https://pytorch.org/get-started/locally/)

To find driver version : Windows Users -

Open Command Prompt and use Command
```
nvidia-smi
```
The driver version should appear near the top right. Ex. (13.1, 12.6)

Install all Python dependencies: 
```
pip install -r requirements.txt
```

Other packages in requirements.txt should install without issues on Python 3.11.

For GPU installations, always check PyTorch Get Started
 for compatible versions.
