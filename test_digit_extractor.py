"""
Unit tests for the Gujarati Digit Extractor

Run with: python -m pytest test_digit_extractor.py
Or simply: python test_digit_extractor.py
"""

import unittest
import os
import shutil
import cv2
import numpy as np
import csv
from digit_extractor import DigitExtractor
from create_sample_image import create_sample_grid_image


class TestDigitExtractor(unittest.TestCase):
    """Test cases for the DigitExtractor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - create test image and output directory."""
        cls.test_image = "test_sample.jpg"
        cls.test_output_dir = "test_output"
        cls.participant_id = "TEST001"
        
        # Create a smaller test image for faster testing
        create_sample_grid_image(cls.test_image, width=1000, height=1000)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files and directories."""
        if os.path.exists(cls.test_image):
            os.remove(cls.test_image)
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
    
    def setUp(self):
        """Set up for each test."""
        # Create fresh output directory for each test
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        os.makedirs(self.test_output_dir)
    
    def test_extractor_initialization(self):
        """Test that the extractor initializes correctly."""
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir
        )
        
        self.assertEqual(extractor.input_image_path, self.test_image)
        self.assertEqual(extractor.participant_id, self.participant_id)
        self.assertEqual(extractor.output_dir, self.test_output_dir)
        self.assertEqual(extractor.target_size, 256)
        self.assertTrue(os.path.exists(self.test_output_dir))
    
    def test_load_image(self):
        """Test that the image loads correctly."""
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir
        )
        
        image = extractor.load_image()
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # Should be color image
        self.assertEqual(image.shape[2], 3)    # Should have 3 channels
    
    def test_full_extraction_process(self):
        """Test the complete extraction process."""
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir,
            target_size=64  # Use smaller size for faster testing
        )
        
        # Run the full process
        extractor.process()
        
        # Check that 100 images were created
        output_files = [f for f in os.listdir(self.test_output_dir) if f.endswith('.png')]
        self.assertEqual(len(output_files), 100, f"Expected 100 images, got {len(output_files)}")
    
    def test_output_image_properties(self):
        """Test that output images have correct properties."""
        target_size = 128
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir,
            target_size=target_size
        )
        
        extractor.process()
        
        # Check a sample output image
        sample_file = os.path.join(self.test_output_dir, f"{self.participant_id}_3_05.png")
        self.assertTrue(os.path.exists(sample_file), f"Sample file {sample_file} not found")
        
        # Load and check image properties
        img = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (target_size, target_size))
        
        # Check that image is grayscale (single channel)
        self.assertEqual(len(img.shape), 2)
    
    def test_metadata_creation(self):
        """Test that metadata.csv is created correctly."""
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir
        )
        
        extractor.process()
        
        # Check metadata file exists
        metadata_file = os.path.join(self.test_output_dir, 'metadata.csv')
        self.assertTrue(os.path.exists(metadata_file))
        
        # Read and verify metadata
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Should have 100 rows (one for each digit)
        self.assertEqual(len(rows), 100)
        
        # Check column names
        self.assertEqual(set(rows[0].keys()), 
                        {'filename', 'participant_id', 'digit_label', 'sample_no'})
        
        # Check a specific row
        row = rows[0]
        self.assertTrue(row['filename'].startswith(self.participant_id))
        self.assertEqual(row['participant_id'], self.participant_id)
    
    def test_filename_format(self):
        """Test that filenames follow the correct format."""
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir
        )
        
        extractor.process()
        
        # Check that all filenames follow the pattern: ParticipantID_DigitLabel_SampleNo.png
        for digit in range(10):
            for sample in range(1, 11):
                expected_filename = f"{self.participant_id}_{digit}_{sample:02d}.png"
                filepath = os.path.join(self.test_output_dir, expected_filename)
                self.assertTrue(os.path.exists(filepath), 
                              f"Expected file {expected_filename} not found")
    
    def test_preprocessing(self):
        """Test that preprocessing works correctly."""
        extractor = DigitExtractor(
            self.test_image,
            self.participant_id,
            self.test_output_dir
        )
        
        # Create a simple test cell image
        cell_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Draw a simple digit (circle) in the center
        cv2.circle(cell_image, (50, 50), 20, (0, 0, 0), 2)
        
        # Preprocess the digit
        processed = extractor.preprocess_digit(cell_image)
        
        # Check that result is the correct size
        self.assertEqual(processed.shape, (extractor.target_size, extractor.target_size))
        
        # Check that it's grayscale
        self.assertEqual(len(processed.shape), 2)
        
        # Check that values are in valid range
        self.assertTrue(processed.min() >= 0)
        self.assertTrue(processed.max() <= 255)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧪 Running Digit Extractor Unit Tests")
    print("="*60 + "\n")
    
    # Run the tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDigitExtractor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("="*60 + "\n")
