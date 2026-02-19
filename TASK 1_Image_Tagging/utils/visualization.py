"""
Visualization Module for Image Classification
Week 1: Image Tagging with TensorFlow

This module provides visualization utilities for training history,
confusion matrix, and model predictions.

Author: Matrix Agent
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras History object from model.fit()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', 
                 linewidth=2, color='blue', marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', 
                 linewidth=2, color='orange', marker='s', markersize=4)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss Plot
    axes[1].plot(history.history['loss'], label='Training Loss', 
                 linewidth=2, color='blue', marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', 
                 linewidth=2, color='orange', marker='s', markersize=4)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], square=True, linewidths=0.5)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return fig, cm


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("=" * 70)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    return report


def plot_sample_images(x, y, class_names, num_samples=25, save_path=None):
    """
    Plot a grid of sample images from the dataset.
    
    Args:
        x: Image data
        y: Labels
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Optional path to save the figure
    """
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.4 * rows))
    axes = axes.flatten()
    
    indices = np.random.choice(len(x), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(x[idx])
        label_idx = y[idx] if y.ndim == 1 else np.argmax(y[idx])
        axes[i].set_title(class_names[label_idx], fontsize=10)
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from CIFAR-10', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("  - plot_training_history(history, save_path=None)")
    print("  - plot_confusion_matrix(y_true, y_pred, class_names, save_path=None)")
    print("  - print_classification_report(y_true, y_pred, class_names)")
    print("  - plot_sample_images(x, y, class_names, num_samples=25)")
