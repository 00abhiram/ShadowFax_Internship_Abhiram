"""
CNN Model Architecture for Image Classification
Week 1: Image Tagging with TensorFlow

This module contains various CNN architectures for image classification tasks.
Supports CIFAR-10 dataset with 10 classes: airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck.

Author: Matrix Agent
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16, ResNet50


class ImageClassifierCNN:
    """
    A flexible CNN-based image classifier supporting multiple architectures.
    
    Attributes:
        input_shape: Tuple defining input image dimensions (height, width, channels)
        num_classes: Number of output classes
        model: The compiled Keras model
    """
    
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        """
        Initialize the ImageClassifierCNN.
        
        Args:
            input_shape: Shape of input images (default: CIFAR-10 size)
            num_classes: Number of classification categories
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self, architecture='standard'):
        """
        Build the CNN model based on specified architecture.
        
        Args:
            architecture: Type of architecture ('simple', 'standard', 'advanced', 'vgg_transfer')
        
        Returns:
            Compiled Keras model
        """
        if architecture == 'simple':
            self.model = self._build_simple_cnn()
        elif architecture == 'standard':
            self.model = self._build_standard_cnn()
        elif architecture == 'advanced':
            self.model = self._build_advanced_cnn()
        elif architecture == 'vgg_transfer':
            self.model = self._build_transfer_learning_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return self.model
    
    def _build_simple_cnn(self):
        """Build a simple CNN with 2 convolutional blocks."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_standard_cnn(self):
        """Build a standard CNN with 3 convolutional blocks and batch normalization."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_advanced_cnn(self):
        """Build an advanced CNN with residual-like connections and regularization."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial Convolution
        x = layers.Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.0001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Block 1
        x = self._residual_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Block 2
        x = self._residual_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Block 3
        x = self._residual_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense Layers
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _residual_block(self, x, filters):
        """Create a residual-like block with skip connection."""
        # Save input for skip connection
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def _build_transfer_learning_model(self):
        """Build a model using VGG16 transfer learning (requires larger input)."""
        # Note: VGG16 requires minimum 32x32 input, but works better with larger images
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(32, 32, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """
        Compile the model with specified optimizer and learning rate.
        
        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate for the optimizer
        """
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model_summary(self):
        """Print model summary."""
        if self.model is not None:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def save_model(self, filepath):
        """Save the model to a file."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")
    
    def load_model(self, filepath):
        """Load a model from a file."""
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Factory function to create a simple CNN model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_advanced_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Factory function to create an advanced CNN model with regularization.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    classifier = ImageClassifierCNN(input_shape=input_shape, num_classes=num_classes)
    model = classifier.build_model(architecture='advanced')
    classifier.compile_model(optimizer='adam', learning_rate=0.001)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN Model Creation...")
    
    classifier = ImageClassifierCNN(input_shape=(32, 32, 3), num_classes=10)
    
    # Test simple architecture
    print("\n=== Simple CNN Architecture ===")
    model = classifier.build_model(architecture='simple')
    classifier.compile_model()
    classifier.get_model_summary()
    
    # Test standard architecture
    print("\n=== Standard CNN Architecture ===")
    model = classifier.build_model(architecture='standard')
    classifier.compile_model()
    classifier.get_model_summary()
    
    # Test advanced architecture
    print("\n=== Advanced CNN Architecture ===")
    model = classifier.build_model(architecture='advanced')
    classifier.compile_model()
    classifier.get_model_summary()
    
    print("\nAll models created successfully!")
