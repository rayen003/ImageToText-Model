# Image Captioning with BERT and VGG16

This project implements an image captioning model using BERT for text embeddings and VGG16 for image feature extraction. The model architecture consists of an encoder and a decoder.

## Table of Contents

- [Introduction](#introduction)
- [Creating a Small Dataset](#creating-a-small-dataset)
- [Preprocessing Pipelines](#preprocessing-pipelines)
  - [Image Preprocessing Pipeline](#image-preprocessing-pipeline)
  - [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Model Architecture](#model-architecture)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
- [Training and Evaluation](#training-and-evaluation)

## Introduction

This project aims to generate captions for images using a combination of BERT for text embeddings and VGG16 for image feature extraction. The model architecture includes an encoder to process image features and a decoder to generate captions.

## Creating a Small Dataset

To make initial experiments easier, we create a small dataset by sampling 1% of the total images and their corresponding captions. This helps in reducing the computational load and speeds up the experimentation process.

### Steps to Create a Small Dataset

1. **Directories and Files**:
   - `Images`: Directory containing the original images.
   - `Small_Images`: Directory to store the sampled images.
   - `captions.txt`: File containing the original captions.
   - `small_captions.txt`: File to store the sampled captions.

2. **Script to Create Small Dataset**:
   - The `data.py` script samples 1% of the images and their corresponding captions and saves them in the `Small_Images` directory and `small_captions.txt` file.


```python
import os
import random
from PIL import Image

# Directories
images_dir = "Images"
small_images_dir = "Small_Images"
captions_file = "captions.txt"
small_captions_file = "small_captions.txt"

# Create the directory for small images if it doesn't exist
os.makedirs(small_images_dir, exist_ok=True)

# List all images in the directory
images = os.listdir(images_dir)

# Calculate 1% of the total images
images_to_keep = max(1, len(images) // 100)

# Select 1% of the images randomly
sampled_images = random.sample(images, images_to_keep)

# Copy the sampled images to the new directory
for image in sampled_images:
    img = Image.open(f'{images_ir}/{image}')
    img.save(f'{small_images_dir}/{image}')

# Read the original captions file
with open(captions_file, 'r') as file:
    lines = file.readlines()

# Filter captions for the sampled images
header = lines[0]  # Keep the header
sampled_captions = [header] + [line for line in lines[1:] if line.split(',')[0] in sampled_images]

# Write the sampled captions to a new file
with open(small_captions_file, 'w') as file:
    file.writelines(sampled_captions)

print(f'Sampled {images_to_keep} images out of {len(images)} total images.')
print(f'Sampled captions written to {small_captions_file}.')
```
Preprocessing Pipelines
Image Preprocessing Pipeline
The image preprocessing pipeline uses VGG16 to extract features from the images. The features are extracted from the fc2 layer of VGG16.

Text Preprocessing Pipeline
The text preprocessing pipeline uses BERT to encode the captions. The BERT tokenizer and model are used to preprocess and encode the captions.

Model Architecture
The model architecture consists of an encoder and a decoder. The encoder processes the image features, and the decoder generates captions.

Encoder
The encoder uses a pre-trained VGG16 model to extract features from the images.


Decoder
The decoder uses BERT embeddings and an LSTM to generate captions.

Training and Evaluation
The model is trained using the preprocessed image features and encoded captions. The training process involves defining a data generator, creating a dataset, and fitting the model.

Main Script
The main.py script integrates the preprocessing pipelines and model architecture to train and evaluate the model.

