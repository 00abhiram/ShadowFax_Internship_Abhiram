"""
Data Loading and Augmentation Module
Week 1: Image Tagging with TensorFlow

This module handles data loading, preprocessing, and augmentation for CIFAR-10.

Author: Matrix Agent
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# CIFAR-10 Class Names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_class_names():
    """Get the class names for CIFAR-10 dataset."""
    return CIFAR10_CLASSES


def load_cifar10_data(normalize=True, one_hot=False):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        normalize: Whether to normalize pixel values to [0, 1]
        one_hot: Whether to one-hot encode labels
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Normalize pixel values
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        print("Data normalized to [0, 1] range")
    
    # One-hot encode labels
    if one_hot:
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        print("Labels one-hot encoded")
    else:
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    
    return (x_train, y_train), (x_test, y_test)


def create_data_augmentation(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True,
                             zoom_range=0.1, validation_split=0.1):
    """
    Create an ImageDataGenerator with data augmentation.
    
    Args:
        rotation_range: Degree range for random rotations
        width_shift_range: Fraction of total width for horizontal shifts
        height_shift_range: Fraction of total height for vertical shifts
        horizontal_flip: Whether to randomly flip images horizontally
        zoom_range: Range for random zoom
        validation_split: Fraction of data to use for validation
    
    Returns:
        Configured ImageDataGenerator instance
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    print("Data augmentation configured:")
    print(f"  - Rotation range: ±{rotation_range}°")
    print(f"  - Width shift: ±{width_shift_range*100}%")
    print(f"  - Height shift: ±{height_shift_range*100}%")
    print(f"  - Horizontal flip: {horizontal_flip}")
    print(f"  - Zoom range: ±{zoom_range*100}%")
    
    return datagen


if __name__ == "__main__":
    print("Testing Data Loading Module")
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(one_hot=True)
    print(f"Training labels shape: {y_train.shape}")
    datagen = create_data_augmentation()
    print("Data loading module test complete!")
