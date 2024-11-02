# Flower Classification using Single-Layer Perceptron

This notebook demonstrates the use of a **single-layer perceptron (SLP)** for binary classification of flower images. The goal is to classify flower images into one of two categories using a basic neural network model trained to recognize pixel patterns associated with each category.

## Overview

The objective of this notebook is to implement a simple approach to image classification using a single-layer perceptron. We train this model on flower images organized into two distinct folders, each representing one category. The model learns to classify the images based on pixel intensity patterns.

## Dataset

The dataset is structured with images separated into two folders, each representing one flower category (e.g., `rose/` and `tulip/`). The directory structure serves as the basis for labeling:
- **rose/** – contains all images of the first flower category.
- **tulip/** – contains all images of the second flower category.

Images are resized and flattened into vectors, making it easier for the perceptron to process them as 1D arrays of pixel intensities.

**Dataset can be found in folder "Datasets" named "flowers"**

## Model Architecture

### Single-Layer Perceptron (SLP)

The model architecture consists of:
- **Input Layer**: Each image is flattened into a one-dimensional vector of pixel values. For example, a 28x28 image would yield 784 features.
- **Output Layer**: A single output neuron with a softmax activation function, providing a probability score for one of the two categories.

This architecture is a minimal setup, enabling the model to learn basic pixel-level patterns while remaining computationally lightweight.

## Training Process

1. **Data Preprocessing**: Images are resized to a consistent dimension and normalized. Each image is flattened to a single vector, which becomes the model input.
2. **Labeling**: Images are assigned labels based on their containing folder (e.g., `rose/` is labeled as 0, and `tulip/` as 1).
3. **Forward Propagation**: The perceptron processes each image to output a probability score for the given category.
4. **Loss Calculation**: The softmax cross-entropy loss is used to measure the difference between predicted probabilities and actual labels:
5. **Optimization**: Weights are adjusted via backpropagation to minimize loss, typically using gradient descent.
6. **Epochs**: Training runs over multiple epochs, gradually improving model performance as weights are optimized.

## Evaluation and Visualization

- **Test Accuracy**: The model's performance is evaluated on a test set, providing a measure of accuracy on unseen data.
- **Loss Curve**: A cost function plot shows the model’s loss over each epoch, visualizing the learning process.
- **Prediction Visualization**: Sample test images are displayed with predicted and actual labels to visually assess model performance.

This notebook provides a hands-on understanding of how single-layer perceptrons work, demonstrating a foundational approach to binary image classification.

