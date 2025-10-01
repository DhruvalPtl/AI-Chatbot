# Gujarati Handwritten Digit Extractor

This script extracts individual digit images from scanned sheets containing handwritten Gujarati digits arranged in a 10×10 grid format.

## Overview

The digit extractor processes scanned sheets where:
- Digits 0-9 are arranged in a 10×10 grid (one row per digit, one column per sample)
- Each sheet is scanned at 600 DPI with dimensions ≈ 4064×4928 px
- Images are in color (24-bit) with black/blue pen or pencil on white paper

## Features

### 1. **Grid Detection & Cell Extraction**
- Automatically detects the 10×10 grid structure
- Crops individual cells while excluding thick borders
- Handles digits that touch grid lines

### 2. **Image Preprocessing**
- Converts to grayscale
- Applies Otsu's binarization for optimal thresholding
- Removes noise and paper texture
- Centers digits with equal padding
- Creates square images

### 3. **Output Format**
- Saves each digit as a separate PNG file
- Default size: 256×256 pixels (configurable)
- Generates metadata CSV file

### 4. **File Naming Convention**
Files are named as: `ParticipantID_DigitLabel_SampleNo.png`

Examples:
- `P001_3_05.png` - Participant P001, digit "3", 5th sample
- `P002_7_02.png` - Participant P002, digit "7", 2nd sample

## Installation

1. Install required dependencies:
```bash
pip install opencv-python-headless numpy
```

Or install all project dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python digit_extractor.py --input P001_test2_cropped.jpg --participant P001
```

### With Custom Output Directory

```bash
python digit_extractor.py --input P001_test2_cropped.jpg --participant P001 --output ./extracted_digits
```

### With Custom Image Size

```bash
python digit_extractor.py --input P001_test2_cropped.jpg --participant P001 --size 64
```

### All Options

```bash
python digit_extractor.py \
    --input P001_test2_cropped.jpg \
    --participant P001 \
    --output ./output \
    --size 256
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Path to input scanned image |
| `--participant` | Yes | - | Participant ID (e.g., P001) |
| `--output` | No | `./output` | Output directory for extracted digits |
| `--size` | No | `256` | Target size for output images (square) |

## Output

The script generates:

1. **Individual digit images** (PNG format)
   - Named as `ParticipantID_DigitLabel_SampleNo.png`
   - Preprocessed and centered
   - Square images at specified size

2. **metadata.csv** with columns:
   - `filename` - Name of the digit image file
   - `participant_id` - Participant identifier
   - `digit_label` - The digit value (0-9)
   - `sample_no` - Sample number (01-10)

### Example Output Structure

```
output/
├── P001_0_01.png
├── P001_0_02.png
├── ...
├── P001_9_09.png
├── P001_9_10.png
└── metadata.csv
```

### Example metadata.csv

```csv
filename,participant_id,digit_label,sample_no
P001_0_01.png,P001,0,01
P001_0_02.png,P001,0,02
P001_3_05.png,P001,3,05
P001_7_09.png,P001,7,09
...
```

## Processing Pipeline

The script follows these steps:

1. **Load Image** - Read the scanned sheet
2. **Grid Detection** - Identify horizontal and vertical grid lines
3. **Cell Extraction** - Extract 10×10 cells with margin to exclude borders
4. **Preprocessing** for each cell:
   - Convert to grayscale
   - Apply Gaussian blur
   - Binarize using Otsu's threshold
   - Remove noise with morphological operations
   - Find digit bounding box
   - Crop with padding
   - Make square by adding equal padding
   - Resize to target size
   - Invert colors (dark digit on white background)
5. **Save** - Write processed images and metadata

## Preprocessing Details

### Binarization
- Uses Otsu's automatic thresholding
- Ensures digit strokes are dark and background is white

### Noise Removal
- Applies morphological closing to fill small gaps
- Applies morphological opening to remove small specks
- Removes paper texture and grid line remnants

### Centering & Padding
- Finds bounding box of digit strokes
- Adds equal padding on all sides
- Creates square image by padding shorter dimension

## Troubleshooting

### Grid Lines Not Detected
If the script cannot detect grid lines properly, it falls back to equal spacing based on image dimensions.

### Poor Quality Digits
- Ensure input image is at least 600 DPI
- Check that the original scan has good contrast
- Verify that digits are clearly visible and not too faint

### Missing Digits
The script processes all 100 cells (10×10 grid). Empty cells will result in mostly white images.

## Integration with Dataset Creation

This script is designed to work as part of a larger dataset creation pipeline:

1. Scan participant sheets
2. Crop/align scans if needed
3. Run this extractor on each sheet
4. Collect all outputs into a dataset directory
5. Use metadata.csv for training/validation splits

## Example Workflow

```bash
# Process multiple participants
python digit_extractor.py --input P001_test2_cropped.jpg --participant P001 --output ./dataset/P001
python digit_extractor.py --input P002_scan.jpg --participant P002 --output ./dataset/P002
python digit_extractor.py --input P003_scan.jpg --participant P003 --output ./dataset/P003

# Combine all metadata
cat ./dataset/*/metadata.csv > combined_metadata.csv
```

## Technical Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Sufficient disk space for output images (100 images × participants × file size)

## License

This script is part of the AI-Chatbot project. See the main repository for license information.
