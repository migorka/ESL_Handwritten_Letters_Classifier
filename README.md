# EMNIST Letters Classification Using Improved LeNet-5 with Pruning

An end-to-end pipeline for recognizing handwritten English letters (A–Z) using the EMNIST Letters dataset and an Improved LeNet-5 CNN with global L1 pruning. The goal is to train a compact, pruned model that learns robust letter representations while reducing the number of effective parameters.

## Overview

This project implements a supervised letter-classification pipeline:

1. Download EMNIST Letters dataset using torchvision
2. Preprocess data
   - augment with random rotation and affine translation
   - normalize to [−1, 1]
3. Define Improved LeNet-5 with BatchNorm and Dropout
4. Apply global L1 unstructured pruning (20% of weights)
5. Train with Adam + CrossEntropyLoss
6. Apply early stopping based on test accuracy
7. Remove pruning reparameterization
8. Evaluate on test set and visualize confusion matrix

## Results & Plots

![Training loss over epochs](plots/training_loss.[Confusion matrix (A–Z)](plots/confusion_matrix.
