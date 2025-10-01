"""
MNIST Dataset Utility Module

This module provides utilities to create, load, and save datasets in MNIST format.
MNIST format specifications:
- Images: 28x28 pixels, grayscale (0-255)
- Labels: Integer values 0-9 (for digits) or any classification labels
- Standard numpy array format
"""

import numpy as np
from PIL import Image
import os
import struct
from typing import Tuple, List, Optional, Union


def preprocess_image_to_mnist(image_path: str) -> np.ndarray:
    """
    Preprocess an image to MNIST format (28x28 grayscale).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: 28x28 grayscale image array with values 0-255
    """
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.uint8)
    
    return img_array


def create_mnist_dataset_from_images(image_paths: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset in MNIST format from a list of images.
    
    Args:
        image_paths: List of paths to image files
        labels: List of corresponding labels
        
    Returns:
        Tuple of (images, labels) as numpy arrays
        - images: Shape (n, 28, 28), dtype uint8
        - labels: Shape (n,), dtype uint8
    """
    if len(image_paths) != len(labels):
        raise ValueError("Number of images must match number of labels")
    
    n_images = len(image_paths)
    images = np.zeros((n_images, 28, 28), dtype=np.uint8)
    
    for i, img_path in enumerate(image_paths):
        images[i] = preprocess_image_to_mnist(img_path)
    
    labels_array = np.array(labels, dtype=np.uint8)
    
    return images, labels_array


def save_mnist_format(images: np.ndarray, labels: np.ndarray, images_file: str, labels_file: str):
    """
    Save dataset in MNIST binary format (IDX file format).
    
    Args:
        images: Image array of shape (n, 28, 28)
        labels: Label array of shape (n,)
        images_file: Output path for images file
        labels_file: Output path for labels file
    """
    # Save images in IDX3 format
    with open(images_file, 'wb') as f:
        # Magic number for images (2051)
        f.write(struct.pack('>I', 2051))
        # Number of images
        f.write(struct.pack('>I', images.shape[0]))
        # Number of rows
        f.write(struct.pack('>I', 28))
        # Number of columns
        f.write(struct.pack('>I', 28))
        # Write image data
        f.write(images.tobytes())
    
    # Save labels in IDX1 format
    with open(labels_file, 'wb') as f:
        # Magic number for labels (2049)
        f.write(struct.pack('>I', 2049))
        # Number of labels
        f.write(struct.pack('>I', labels.shape[0]))
        # Write label data
        f.write(labels.tobytes())


def load_mnist_format(images_file: str, labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from MNIST binary format (IDX file format).
    
    Args:
        images_file: Path to images file
        labels_file: Path to labels file
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    # Load images
    with open(images_file, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in images file")
        
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
    
    # Load labels
    with open(labels_file, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in labels file")
        
        n_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels


def save_mnist_numpy(images: np.ndarray, labels: np.ndarray, output_dir: str, prefix: str = "mnist"):
    """
    Save dataset in numpy format (.npy files).
    
    Args:
        images: Image array of shape (n, 28, 28)
        labels: Label array of shape (n,)
        output_dir: Directory to save files
        prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images_path = os.path.join(output_dir, f"{prefix}_images.npy")
    labels_path = os.path.join(output_dir, f"{prefix}_labels.npy")
    
    np.save(images_path, images)
    np.save(labels_path, labels)
    
    print(f"Saved images to: {images_path}")
    print(f"Saved labels to: {labels_path}")


def load_mnist_numpy(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from numpy format (.npy files).
    
    Args:
        images_path: Path to images .npy file
        labels_path: Path to labels .npy file
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    images = np.load(images_path)
    labels = np.load(labels_path)
    
    return images, labels


def normalize_mnist_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize MNIST images to 0-1 range.
    
    Args:
        images: Image array with values 0-255
        
    Returns:
        Normalized image array with values 0-1
    """
    return images.astype(np.float32) / 255.0


def get_mnist_stats(images: np.ndarray, labels: np.ndarray) -> dict:
    """
    Get statistics about an MNIST dataset.
    
    Args:
        images: Image array
        labels: Label array
        
    Returns:
        Dictionary with dataset statistics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats = {
        'n_samples': len(images),
        'image_shape': images.shape[1:],
        'n_classes': len(unique_labels),
        'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
        'min_pixel_value': images.min(),
        'max_pixel_value': images.max(),
        'mean_pixel_value': images.mean(),
    }
    
    return stats
