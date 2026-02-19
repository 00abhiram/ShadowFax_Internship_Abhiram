"""
Main Training Script for CIFAR-10 Image Classification
Week 1: Image Tagging with TensorFlow - ShadowFox AIML Internship

This script performs:
1. Data Loading & Preprocessing (Normalization + One-Hot Encoding)
2. Data Augmentation (rotation, flip, zoom)
3. CNN Model Training with Adam optimizer
4. Evaluation with Confusion Matrix & Classification Report

Author: ShadowFox AIML Intern
Date: February 2026
"""

import os
import sys
import numpy as np
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from project modules
from models.cnn_model import ImageClassifierCNN
from utils.data_loader import load_cifar10_data, create_data_augmentation, get_class_names
from utils.visualization import (plot_training_history, plot_confusion_matrix, 
                                  print_classification_report, plot_sample_images)

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks():
    """Create training callbacks."""
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
                       monitor='val_accuracy', save_best_only=True, verbose=1)
    ]


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main function to run the complete training pipeline."""
    
    print("\n" + "=" * 70)
    print("CIFAR-10 IMAGE CLASSIFICATION - WEEK 1 SHADOWFOX INTERNSHIP")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus)} device(s)" if gpus else "Using CPU")
    
    # Get class names
    class_names = get_class_names()
    
    # ========================================================================
    # STEP 1: Load and Preprocess Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(normalize=True, one_hot=True)
    
    print(f"\nTraining samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Labels shape: {y_train.shape}")
    
    # ========================================================================
    # STEP 2: Data Augmentation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: CONFIGURING DATA AUGMENTATION")
    print("=" * 70)
    
    datagen = create_data_augmentation(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=VALIDATION_SPLIT
    )
    datagen.fit(x_train)
    
    # ========================================================================
    # STEP 3: Build and Compile Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: BUILDING AND COMPILING CNN MODEL")
    print("=" * 70)
    
    classifier = ImageClassifierCNN(input_shape=(32, 32, 3), num_classes=10)
    model = classifier.build_model(architecture='standard')
    
    # Compile with Adam optimizer and categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nOptimizer: Adam (lr={LEARNING_RATE})")
    print(f"Loss: categorical_crossentropy")
    print("\nModel Architecture:")
    model.summary()
    
    # ========================================================================
    # STEP 4: Train Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING THE MODEL")
    print("=" * 70)
    
    train_samples = int(len(x_train) * (1 - VALIDATION_SPLIT))
    val_samples = int(len(x_train) * VALIDATION_SPLIT)
    steps_per_epoch = train_samples // BATCH_SIZE
    validation_steps = val_samples // BATCH_SIZE
    
    print(f"\nEpochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, subset='training'),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, subset='validation'),
        validation_steps=validation_steps,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # ========================================================================
    # STEP 5: Plot Training History
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING ACCURACY VS LOSS PLOT")
    print("=" * 70)
    
    history_plot_path = os.path.join(OUTPUT_DIR, 'accuracy_loss_plot.png')
    plot_training_history(history, save_path=history_plot_path)
    
    # ========================================================================
    # STEP 6: Evaluate Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATING MODEL ON TEST DATA")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # Get predictions
    y_pred_probs = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # ========================================================================
    # STEP 7: Confusion Matrix
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING CONFUSION MATRIX")
    print("=" * 70)
    
    cm_plot_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_plot_path)
    
    # ========================================================================
    # STEP 8: Classification Report
    # ========================================================================
    print_classification_report(y_true, y_pred, class_names)
    
    # ========================================================================
    # Save Final Model
    # ========================================================================
    final_model_path = os.path.join(MODEL_DIR, 'final_model.keras')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"\nOutput Files:")
    print(f"  - Accuracy/Loss Plot: {history_plot_path}")
    print(f"  - Confusion Matrix: {cm_plot_path}")
    print(f"  - Best Model: {os.path.join(MODEL_DIR, 'best_model.keras')}")
    print(f"  - Final Model: {final_model_path}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
