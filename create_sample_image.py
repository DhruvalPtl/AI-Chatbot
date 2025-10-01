"""
Sample Test Image Generator

This script creates a synthetic test image that mimics the structure of a 
handwritten digit sheet for testing the digit extractor.
"""

import cv2
import numpy as np
import os


def create_sample_grid_image(output_path="P001_test2_cropped.jpg", width=4064, height=4928):
    """
    Create a sample 10x10 grid image with digits for testing.
    
    Args:
        output_path: Path to save the generated test image
        width: Width of the output image
        height: Height of the output image
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Calculate cell dimensions
    cell_width = width // 10
    cell_height = height // 10
    
    # Draw grid lines
    line_thickness = 3
    line_color = (0, 0, 0)  # Black
    
    # Draw vertical lines
    for i in range(11):
        x = i * cell_width
        cv2.line(image, (x, 0), (x, height), line_color, line_thickness)
    
    # Draw horizontal lines
    for i in range(11):
        y = i * cell_height
        cv2.line(image, (0, y), (width, y), line_color, line_thickness)
    
    # Add digit text in each cell
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 8
    font_thickness = 15
    text_color = (0, 0, 255)  # Blue (BGR format)
    
    for row in range(10):
        digit = row  # Digit label is the row number (0-9)
        for col in range(10):
            # Calculate text position (centered in cell)
            text = str(digit)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Center of the cell
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            
            # Text position (bottom-left corner)
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            
            # Add some random variation to simulate handwriting
            offset_x = np.random.randint(-30, 30)
            offset_y = np.random.randint(-30, 30)
            text_x += offset_x
            text_y += offset_y
            
            # Draw the digit
            cv2.putText(image, text, (text_x, text_y), font, font_scale, 
                       text_color, font_thickness, cv2.LINE_AA)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"✓ Sample test image created: {output_path}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Grid: 10x10 cells")
    print(f"  Cell size: {cell_width}x{cell_height}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate a sample test image for digit extractor'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='P001_test2_cropped.jpg',
        help='Output path for the test image'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=4064,
        help='Width of the test image (default: 4064)'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=4928,
        help='Height of the test image (default: 4928)'
    )
    
    args = parser.parse_args()
    
    create_sample_grid_image(args.output, args.width, args.height)
