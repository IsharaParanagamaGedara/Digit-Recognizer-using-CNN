# Digit Recognizer using CNN

This project implements a Convolutional Neural Network (CNN) model to recognize handwritten digits (0-9) from the MNIST dataset. The model is trained using the `train.csv` dataset and makes predictions on the `test.csv` dataset. The project is designed using Python and deep learning libraries such as TensorFlow and Keras.

## Table of Contents
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Project Overview](#project-overview)
  - [Data Preprocessing](#data-preprocessing)
  - [CNN Model Architecture](#cnn-model-architecture)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Submission](#submission)
- [Conclusion](#conclusion)

## Dataset
The project uses the MNIST dataset, which contains images of handwritten digits.

- **Training data**: `train.csv` contains 42,000 images, each represented by 784 pixel values (28x28 images) and a label indicating the digit.
- **Test data**: `test.csv` contains 28,000 images without labels.
- **Sample Submission**: `sample_submission.csv` is used to create predictions for the test dataset.

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow.keras`
- `warnings`

## Project Overview

### Data Preprocessing
- The data is read from `train.csv` and `test.csv`.
- The pixel values are normalized between 0 and 1.
- The training data is split into training and validation sets.
- Data augmentation techniques such as rotation, width/height shifting, and zooming are used to increase the dataset's diversity.

### CNN Model Architecture
The CNN model consists of the following layers:
1. Three Convolutional Layers with ReLU activation and Max Pooling for feature extraction.
2. Flatten Layer to convert the feature maps into a 1D vector.
3. Dense layers for classification, with Dropout for regularization.
4. Final output layer with softmax activation for digit classification.

### Model Training
- The model is compiled using Adam optimizer and sparse categorical cross-entropy loss.
- The model is trained for 20 epochs with a batch size of 64 using data augmentation.
- Validation accuracy and loss are monitored during training.

### Evaluation
- Confusion matrix and classification report are used to evaluate the model's performance on the validation set.
- The model achieves an accuracy of around 98% on the validation set.

## Results
- **Model Accuracy**: 98.68% on the validation set.
- **Model Loss**: Validation loss decreases to 0.1505.
- The model's performance is visualized using confusion matrix and classification reports, and predictions are made on the test set.

## Submission
The predicted results are saved in `submission.csv`, which contains the image ID and the corresponding predicted digit.

## Conclusion
This project successfully implements a CNN-based digit recognizer that performs well on the MNIST dataset with high accuracy. The model could be further improved by fine-tuning the architecture and training for more epochs.

