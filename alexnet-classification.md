
# House Room Classification with AlexNet

This notebook aims to classify images of different types of rooms in a house (e.g., Bathroom, Bedroom, Dining Room, Living Room and kitchen) using a Convolutional Neural Network (CNN) based on the AlexNet architecture. The dataset contains labeled images organized by class folders, which are preprocessed and augmented using TensorFlow's `ImageDataGenerator`.

## Project Structure

- **Dataset**: The dataset is organized in folders for each room type:
  ```bash
  /House_Room_Dataset
  ├── Bathroom
  ├── Bedroom
  ├── Dining
  ├── Livingroom
  ├── Kitchen
  ```
  Link : https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset/data

- **Notebook**: The notebook follows a structured pipeline:
  1. **Importing Libraries**: Imports required libraries for deep learning, data preprocessing, and visualization.
  2. **Data Preprocessing and Augmentation**: Utilizes `ImageDataGenerator` to prepare the data.
  3. **AlexNet Model Creation**: Builds the AlexNet model architecture in TensorFlow.
  4. **Training the Model**: Compiles and trains the model on the dataset.
  5. **Evaluation**: Plots training and validation metrics to evaluate model performance.
  6. **Saving the Model**: Saves the trained model for future use.

## AlexNet Architecture
![Alexnet](https://github.com/user-attachments/assets/2afda91c-0047-4b75-9c69-3e4e6a77dcb5)

In 2012, image recognition saw its blurry world come into razor-sharp focus with the arrival of AlexNet, a game-changing deep neural network. 

**AlexNet** is a popular CNN architecture known for its deep and powerful feature extraction capabilities.

 `AlexNet consists of 5 convolution layers, 3 max-pooling layers, 2 Normalized layers, 2 fully connected layers and 1 SoftMax layer.`

1. **Convolutional Layers**: Extract features by detecting patterns in the input images.
2. **Max Pooling Layers**: Reduce the spatial dimensions, retaining essential information and reducing computation.
3. **Fully Connected Layers**: Process the flattened output for classification.
4. **Dropout Layers**: Prevent overfitting by randomly dropping units during training.

AlexNet was designed for large-scale image classification and can handle high-dimensional images, making it a suitable choice for complex image datasets.

Paper link : https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

### Model Summary

```plaintext
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 55, 55, 96)        34944     
max_pooling2d (MaxPooling2D) (None, 27, 27, 96)        0         
... (additional layers) ...
dense_2 (Dense)              (None, 4)                 40964     
=================================================================
Total params: 61,100,840
Trainable params: 61,100,840
Non-trainable params: 0
```

## Data Preprocessing and Augmentation with `ImageDataGenerator`

To enhance model generalization, **data augmentation** techniques are applied, making the model robust to variations in data and preventing overfitting.

```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
```

- **Rescale**: Normalizes pixel values to a range of `[0, 1]` for faster model convergence.
- **Rotation, Width/Height Shifts, Shear, Zoom, Flip**: Adds diversity to the images by altering their orientation and size.
- **Validation Split**: Automatically splits 80% of the data for training and 20% for validation, ensuring the model is evaluated on unseen data.

### Training and Validation Generators

Using `ImageDataGenerator.flow_from_directory()`, we create generators for training and validation data. These generators load images in batches, preprocess them on-the-fly, and feed them to the model.

```python
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

### Training the Model

The model is compiled with the Adam optimizer and trained on the processed data. Both training and validation performance metrics are recorded for each epoch to monitor the model's learning progress.

## Evaluation and Results

After training, the model's accuracy and loss are visualized over epochs to assess convergence and generalization. This helps in understanding if the model is overfitting or if it requires further tuning.



## Requirements

- Python 3.x
- TensorFlow
- Keras
- Matplotlib



## Conclusion

This notebook demonstrates how to implement AlexNet for multi-class image classification using TensorFlow.
