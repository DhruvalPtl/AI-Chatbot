# MNIST Dataset Utilities

This module provides utilities to create, load, and save datasets in MNIST format.

## Overview

The MNIST format is a standard format for image datasets, originally created for the famous handwritten digit dataset. This utility allows you to:
- Convert your own images to MNIST format (28x28 grayscale)
- Save datasets in MNIST binary format (IDX files)
- Save datasets in NumPy format
- Load MNIST-formatted datasets
- Get statistics about your dataset
- Normalize images for machine learning

## MNIST Format Specifications

- **Images**: 28x28 pixels, grayscale (values 0-255)
- **Labels**: Integer values (0-9 for digits, or any classification labels)
- **File Format**: IDX binary format or NumPy arrays

## Usage Examples

### 1. Creating MNIST Dataset from Images

```python
from mnist_dataset_utils import create_mnist_dataset_from_images

# Prepare your image paths and labels
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
labels = [0, 1, 2, ...]  # Corresponding labels

# Create MNIST-formatted dataset
images, labels = create_mnist_dataset_from_images(image_paths, labels)
# Returns: images (n, 28, 28), labels (n,)
```

### 2. Saving in MNIST Binary Format (IDX)

```python
from mnist_dataset_utils import save_mnist_format

# Save in IDX format (original MNIST format)
save_mnist_format(images, labels, 'images.idx', 'labels.idx')
```

### 3. Saving in NumPy Format

```python
from mnist_dataset_utils import save_mnist_numpy

# Save in NumPy format (easier to work with in Python)
save_mnist_numpy(images, labels, output_dir='./data', prefix='my_dataset')
# Creates: ./data/my_dataset_images.npy and ./data/my_dataset_labels.npy
```

### 4. Loading Datasets

```python
from mnist_dataset_utils import load_mnist_format, load_mnist_numpy

# Load from IDX format
images, labels = load_mnist_format('images.idx', 'labels.idx')

# Or load from NumPy format
images, labels = load_mnist_numpy('images.npy', 'labels.npy')
```

### 5. Getting Dataset Statistics

```python
from mnist_dataset_utils import get_mnist_stats

stats = get_mnist_stats(images, labels)
print(f"Number of samples: {stats['n_samples']}")
print(f"Number of classes: {stats['n_classes']}")
print(f"Class distribution: {stats['class_distribution']}")
```

### 6. Normalizing Images

```python
from mnist_dataset_utils import normalize_mnist_images

# Normalize from 0-255 to 0-1 range
normalized_images = normalize_mnist_images(images)
```

## Available Functions

### `preprocess_image_to_mnist(image_path: str) -> np.ndarray`
Preprocess a single image to MNIST format (28x28 grayscale).

### `create_mnist_dataset_from_images(image_paths: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]`
Create a complete MNIST dataset from a list of image files.

### `save_mnist_format(images: np.ndarray, labels: np.ndarray, images_file: str, labels_file: str)`
Save dataset in MNIST binary format (IDX files).

### `load_mnist_format(images_file: str, labels_file: str) -> Tuple[np.ndarray, np.ndarray]`
Load dataset from MNIST binary format.

### `save_mnist_numpy(images: np.ndarray, labels: np.ndarray, output_dir: str, prefix: str = "mnist")`
Save dataset in NumPy format (.npy files).

### `load_mnist_numpy(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]`
Load dataset from NumPy format.

### `normalize_mnist_images(images: np.ndarray) -> np.ndarray`
Normalize MNIST images to 0-1 range.

### `get_mnist_stats(images: np.ndarray, labels: np.ndarray) -> dict`
Get comprehensive statistics about a dataset.

## Running the Examples

To see all features in action, run the example script:

```bash
python mnist_example.py
```

This will demonstrate:
- Creating synthetic MNIST-like datasets
- Saving and loading in both IDX and NumPy formats
- Getting dataset statistics
- Normalizing images

## Use Cases

1. **Machine Learning Projects**: Prepare custom image datasets in a standard format
2. **Data Preprocessing**: Convert images to a consistent 28x28 grayscale format
3. **Dataset Creation**: Build custom datasets similar to MNIST for experiments
4. **Data Exchange**: Share datasets in a standardized format
5. **Model Training**: Prepare data in the format expected by many ML models

## Requirements

- numpy
- PIL (Pillow)

These are already included in the project's requirements.txt.
