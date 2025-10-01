"""
Example script demonstrating how to use the MNIST dataset utilities.

This script shows how to:
1. Create MNIST-format datasets from images
2. Save datasets in MNIST format
3. Load MNIST datasets
4. Get dataset statistics
"""

import numpy as np
from mnist_dataset_utils import (
    create_mnist_dataset_from_images,
    save_mnist_format,
    load_mnist_format,
    save_mnist_numpy,
    load_mnist_numpy,
    get_mnist_stats,
    normalize_mnist_images
)


def example_create_dataset():
    """Example: Create a simple MNIST-like dataset"""
    print("=" * 50)
    print("Example: Creating MNIST-like dataset from scratch")
    print("=" * 50)
    
    # For demonstration, we'll create synthetic data instead of loading images
    # In practice, you would use actual image paths
    n_samples = 100
    images = np.random.randint(0, 256, size=(n_samples, 28, 28), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=n_samples, dtype=np.uint8)
    
    print(f"Created dataset with {n_samples} samples")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return images, labels


def example_save_and_load_idx():
    """Example: Save and load in IDX format"""
    print("\n" + "=" * 50)
    print("Example: Save and load in IDX format")
    print("=" * 50)
    
    # Create sample data
    images, labels = example_create_dataset()
    
    # Save in IDX format
    images_file = "/tmp/mnist_images.idx"
    labels_file = "/tmp/mnist_labels.idx"
    
    save_mnist_format(images, labels, images_file, labels_file)
    print(f"\nSaved to IDX format:")
    print(f"  Images: {images_file}")
    print(f"  Labels: {labels_file}")
    
    # Load back
    loaded_images, loaded_labels = load_mnist_format(images_file, labels_file)
    print(f"\nLoaded from IDX format:")
    print(f"  Images shape: {loaded_images.shape}")
    print(f"  Labels shape: {loaded_labels.shape}")
    
    # Verify data integrity
    assert np.array_equal(images, loaded_images), "Images don't match!"
    assert np.array_equal(labels, loaded_labels), "Labels don't match!"
    print("✓ Data integrity verified!")
    
    return loaded_images, loaded_labels


def example_save_and_load_numpy():
    """Example: Save and load in NumPy format"""
    print("\n" + "=" * 50)
    print("Example: Save and load in NumPy format")
    print("=" * 50)
    
    # Create sample data
    images, labels = example_create_dataset()
    
    # Save in NumPy format
    output_dir = "/tmp/mnist_data"
    save_mnist_numpy(images, labels, output_dir, prefix="example")
    
    # Load back
    images_path = f"{output_dir}/example_images.npy"
    labels_path = f"{output_dir}/example_labels.npy"
    
    loaded_images, loaded_labels = load_mnist_numpy(images_path, labels_path)
    print(f"\nLoaded from NumPy format:")
    print(f"  Images shape: {loaded_images.shape}")
    print(f"  Labels shape: {loaded_labels.shape}")
    
    return loaded_images, loaded_labels


def example_dataset_statistics():
    """Example: Get dataset statistics"""
    print("\n" + "=" * 50)
    print("Example: Dataset statistics")
    print("=" * 50)
    
    # Create sample data with known distribution
    images = np.random.randint(0, 256, size=(100, 28, 28), dtype=np.uint8)
    labels = np.array([i % 10 for i in range(100)], dtype=np.uint8)
    
    stats = get_mnist_stats(images, labels)
    
    print("\nDataset Statistics:")
    print(f"  Number of samples: {stats['n_samples']}")
    print(f"  Image shape: {stats['image_shape']}")
    print(f"  Number of classes: {stats['n_classes']}")
    print(f"  Pixel value range: [{stats['min_pixel_value']}, {stats['max_pixel_value']}]")
    print(f"  Mean pixel value: {stats['mean_pixel_value']:.2f}")
    print(f"\n  Class distribution:")
    for label, count in sorted(stats['class_distribution'].items()):
        print(f"    Class {label}: {count} samples")


def example_normalization():
    """Example: Normalize images"""
    print("\n" + "=" * 50)
    print("Example: Image normalization")
    print("=" * 50)
    
    # Create sample data
    images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    
    print(f"Original images:")
    print(f"  Data type: {images.dtype}")
    print(f"  Value range: [{images.min()}, {images.max()}]")
    
    # Normalize
    normalized = normalize_mnist_images(images)
    
    print(f"\nNormalized images:")
    print(f"  Data type: {normalized.dtype}")
    print(f"  Value range: [{normalized.min():.3f}, {normalized.max():.3f}]")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MNIST Dataset Utilities - Examples")
    print("=" * 50)
    
    # Run examples
    example_create_dataset()
    example_save_and_load_idx()
    example_save_and_load_numpy()
    example_dataset_statistics()
    example_normalization()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50 + "\n")
