"""
Gujarati Handwritten Digit Extractor

This script extracts individual digit images from a scanned sheet containing
handwritten digits arranged in a 10×10 grid (10 rows, 10 columns).

Usage:
    python digit_extractor.py --input P001_test2_cropped.jpg --participant P001 --output ./output
"""

import cv2
import numpy as np
import os
import csv
import argparse
from pathlib import Path


class DigitExtractor:
    """Extract and preprocess handwritten digits from a scanned grid sheet."""
    
    def __init__(self, input_image_path, participant_id, output_dir="./output", target_size=256):
        """
        Initialize the DigitExtractor.
        
        Args:
            input_image_path: Path to the input scanned image
            participant_id: Participant ID (e.g., 'P001')
            output_dir: Directory to save extracted digits
            target_size: Target size for output images (default: 256x256)
        """
        self.input_image_path = input_image_path
        self.participant_id = participant_id
        self.output_dir = output_dir
        self.target_size = target_size
        self.metadata = []
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_image(self):
        """Load the input image."""
        self.image = cv2.imread(self.input_image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {self.input_image_path}")
        print(f"Loaded image with shape: {self.image.shape}")
        return self.image
    
    def preprocess_for_grid_detection(self):
        """Preprocess image for grid detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return gray, binary
    
    def detect_grid_lines(self, binary):
        """Detect horizontal and vertical grid lines."""
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines
        grid_lines = cv2.add(horizontal_lines, vertical_lines)
        
        return horizontal_lines, vertical_lines, grid_lines
    
    def find_grid_cells(self, horizontal_lines, vertical_lines):
        """Find cell boundaries from grid lines."""
        # Find horizontal line positions
        h_lines = []
        for i in range(horizontal_lines.shape[0]):
            if np.sum(horizontal_lines[i, :]) > horizontal_lines.shape[1] * 50:
                h_lines.append(i)
        
        # Find vertical line positions
        v_lines = []
        for j in range(horizontal_lines.shape[1]):
            if np.sum(vertical_lines[:, j]) > vertical_lines.shape[0] * 50:
                v_lines.append(j)
        
        # Remove duplicate lines (keep only significant gaps)
        def filter_lines(lines, min_gap=30):
            if not lines:
                return []
            filtered = [lines[0]]
            for line in lines[1:]:
                if line - filtered[-1] > min_gap:
                    filtered.append(line)
            return filtered
        
        h_lines = filter_lines(h_lines)
        v_lines = filter_lines(v_lines)
        
        print(f"Detected {len(h_lines)} horizontal lines and {len(v_lines)} vertical lines")
        
        return h_lines, v_lines
    
    def extract_cells_with_margin(self, h_lines, v_lines, margin=5):
        """Extract cells from the grid with margin to exclude borders."""
        cells = []
        
        # We need 11 lines to create 10 cells
        if len(h_lines) < 11 or len(v_lines) < 11:
            print(f"Warning: Expected 11 lines each, got {len(h_lines)} horizontal and {len(v_lines)} vertical")
            # Use equal spacing if lines not detected properly
            h, w = self.image.shape[:2]
            h_lines = [int(i * h / 10) for i in range(11)]
            v_lines = [int(j * w / 10) for j in range(11)]
        
        # Extract each cell (10x10 grid)
        for row in range(10):
            for col in range(10):
                y1 = h_lines[row] + margin
                y2 = h_lines[row + 1] - margin
                x1 = v_lines[col] + margin
                x2 = v_lines[col + 1] - margin
                
                # Ensure valid coordinates
                if y2 > y1 and x2 > x1:
                    cell = {
                        'row': row,
                        'col': col,
                        'y1': y1,
                        'y2': y2,
                        'x1': x1,
                        'x2': x2,
                        'digit_label': row  # Row index is the digit label (0-9)
                    }
                    cells.append(cell)
        
        print(f"Extracted {len(cells)} cells")
        return cells
    
    def preprocess_digit(self, cell_image):
        """Preprocess a single digit image."""
        # Convert to grayscale if not already
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Binarize using Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal using morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find bounding box of the digit
        coords = cv2.findNonZero(binary)
        
        if coords is not None and len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Crop to bounding box with some padding
            padding = 10
            y1 = max(0, y - padding)
            y2 = min(binary.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(binary.shape[1], x + w + padding)
            
            cropped = binary[y1:y2, x1:x2]
        else:
            # If no content found, use the whole cell
            cropped = binary
        
        # Make it square by adding equal padding
        h, w = cropped.shape
        if h > w:
            diff = h - w
            pad_left = diff // 2
            pad_right = diff - pad_left
            cropped = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right, 
                                        cv2.BORDER_CONSTANT, value=0)
        elif w > h:
            diff = w - h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=0)
        
        # Resize to target size
        resized = cv2.resize(cropped, (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_AREA)
        
        # Invert so digit is dark on white background
        final = cv2.bitwise_not(resized)
        
        return final
    
    def extract_and_save_digits(self, cells):
        """Extract and save all digits from cells."""
        for cell in cells:
            # Extract cell from original image
            cell_image = self.image[cell['y1']:cell['y2'], cell['x1']:cell['x2']]
            
            # Preprocess the digit
            processed_digit = self.preprocess_digit(cell_image)
            
            # Generate filename
            digit_label = cell['digit_label']
            sample_no = cell['col'] + 1  # Column index + 1 (1-10)
            filename = f"{self.participant_id}_{digit_label}_{sample_no:02d}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, processed_digit)
            
            # Add to metadata
            self.metadata.append({
                'filename': filename,
                'participant_id': self.participant_id,
                'digit_label': digit_label,
                'sample_no': f"{sample_no:02d}"
            })
        
        print(f"Saved {len(cells)} digit images to {self.output_dir}")
    
    def save_metadata(self):
        """Save metadata to CSV file."""
        metadata_path = os.path.join(self.output_dir, 'metadata.csv')
        
        with open(metadata_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'participant_id', 'digit_label', 'sample_no']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in self.metadata:
                writer.writerow(row)
        
        print(f"Saved metadata to {metadata_path}")
    
    def process(self):
        """Main processing pipeline."""
        print(f"Starting digit extraction for participant {self.participant_id}")
        
        # Load image
        self.load_image()
        
        # Preprocess for grid detection
        gray, binary = self.preprocess_for_grid_detection()
        
        # Detect grid lines
        h_lines_img, v_lines_img, grid_lines = self.detect_grid_lines(binary)
        
        # Find grid cell boundaries
        h_lines, v_lines = self.find_grid_cells(h_lines_img, v_lines_img)
        
        # Extract cells
        cells = self.extract_cells_with_margin(h_lines, v_lines, margin=5)
        
        # Extract and save digits
        self.extract_and_save_digits(cells)
        
        # Save metadata
        self.save_metadata()
        
        print("✓ Digit extraction completed successfully!")


def main():
    """Main function to run the digit extractor."""
    parser = argparse.ArgumentParser(
        description='Extract Gujarati handwritten digits from a scanned grid sheet'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input scanned image (e.g., P001_test2_cropped.jpg)'
    )
    parser.add_argument(
        '--participant', 
        type=str, 
        required=True,
        help='Participant ID (e.g., P001)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='./output',
        help='Output directory for extracted digits (default: ./output)'
    )
    parser.add_argument(
        '--size', 
        type=int, 
        default=256,
        help='Target size for output images (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Create extractor and process
    extractor = DigitExtractor(
        input_image_path=args.input,
        participant_id=args.participant,
        output_dir=args.output,
        target_size=args.size
    )
    
    extractor.process()


if __name__ == '__main__':
    main()
