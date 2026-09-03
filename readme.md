# IMG2GPS

IMG2GPS is a deep learning image geolocation project developed for CIS 5190: Applied Machine Learning at the University of Pennsylvania.

Given an input photograph, the model predicts the latitude and longitude coordinates where the image was taken.

## Overview

The project uses a ResNet50-based regression model initialized with ImageNet pretrained weights. Instead of directly predicting latitude and longitude, GPS coordinates are transformed into a local Cartesian coordinate system and normalized before training.

The final model combines transfer learning, data augmentation, coordinate normalization, and a blended Huber + Haversine loss function.

## Results

| Model | Validation Distance | Hidden Leaderboard Distance |
| --- | ---: | ---: |
| Baseline Model | ~100 m | ~120 m |
| Initial ResNet50 | ~80 m | ~95 m |
| Augmented ResNet50 | ~45 m | ~82 m |
| **Final Model** | **~30 m** | **~76 m** |

## Model

- ResNet50 backbone pretrained on ImageNet
- Regression head: `2048 → 1024 → 256 → 2`
- Local Cartesian coordinate prediction
- Coordinate normalization
- Huber + Haversine loss
- Staged transfer learning and fine-tuning

## Data Augmentation

The final training pipeline included:

- Random resized cropping
- Horizontal flipping
- Color jitter
- Random perspective distortion
- Random erasing

## Training Optimizations

To improve training efficiency, the pipeline also used:

- In-memory image caching
- `bfloat16` mixed precision training
- Channels-last memory layout
- TF32 matrix multiplication
- `torch.compile`

## Repository Contents

- `IMG2GPS.ipynb` — training, experimentation, and evaluation notebook
- `model.py` — model implementation
- `preprocess.py` — preprocessing and coordinate transformation utilities
- `IMG2GPS.pdf` — final project report

## Dataset

The project was trained on a custom dataset of approximately 1,115 geotagged images collected from mobile devices.

The dataset was also uploaded to Hugging Face for reproducibility and accessibility.