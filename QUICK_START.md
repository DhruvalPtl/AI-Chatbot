# Quick Start Guide - Gujarati Digit Extractor

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install opencv-python-headless numpy
```

### Step 2: Run the Complete Workflow
```bash
# Option A: Use the example workflow script (recommended)
python example_workflow.py

# Option B: Manual steps
# 1. Create test image
python create_sample_image.py --output P001_test2_cropped.jpg

# 2. Extract digits
python digit_extractor.py --input P001_test2_cropped.jpg --participant P001

# 3. Verify output
ls -lh output/
cat output/metadata.csv
```

### Step 3: View Results
```bash
# Count extracted images
ls output/*.png | wc -l
# Should output: 100

# View metadata
head output/metadata.csv
```

## 📝 Using Your Own Scanned Images

1. Place your scanned image in the repository directory
2. Run the extractor:
   ```bash
   python digit_extractor.py --input YOUR_IMAGE.jpg --participant YOUR_ID
   ```

## ⚙️ Advanced Options

### Custom Output Size (e.g., 64×64 for ML)
```bash
python digit_extractor.py \
    --input P001_test2_cropped.jpg \
    --participant P001 \
    --size 64 \
    --output ./digits_64
```

### Process Multiple Participants
```bash
for participant in P001 P002 P003; do
    python digit_extractor.py \
        --input ${participant}_scan.jpg \
        --participant $participant \
        --output ./dataset/$participant
done
```

## 🧪 Run Tests
```bash
python test_digit_extractor.py
```

## 📚 Full Documentation
See [DIGIT_EXTRACTOR_README.md](DIGIT_EXTRACTOR_README.md) for complete documentation.
