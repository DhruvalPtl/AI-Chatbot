#!/usr/bin/env python3
"""
Example: Complete workflow for digit extraction

This script demonstrates the complete workflow:
1. Creating a sample test image (if needed)
2. Running the digit extractor
3. Verifying the output
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}", file=sys.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    else:
        print(f"✓ Success")
        return True


def main():
    """Run the complete example workflow."""
    print("\n" + "="*60)
    print("🎯 Gujarati Handwritten Digit Extraction - Example Workflow")
    print("="*60)
    
    # Step 1: Create sample image (if it doesn't exist)
    sample_image = "P001_test2_cropped.jpg"
    if not os.path.exists(sample_image):
        print(f"\n📝 Step 1: Creating sample test image...")
        if not run_command(
            f"python create_sample_image.py --output {sample_image}",
            "Generate synthetic test image"
        ):
            return
    else:
        print(f"\n✓ Sample image already exists: {sample_image}")
    
    # Step 2: Run digit extractor
    print(f"\n📝 Step 2: Extracting digits...")
    if not run_command(
        f"python digit_extractor.py --input {sample_image} --participant P001 --output ./output",
        "Extract digits from the scanned sheet"
    ):
        return
    
    # Step 3: Verify output
    print(f"\n📝 Step 3: Verifying output...")
    
    # Count output files
    run_command(
        "ls -1 output/*.png | wc -l",
        "Count extracted digit images"
    )
    
    # Show sample files
    run_command(
        "ls -lh output/ | head -15",
        "List first 15 output files"
    )
    
    # Show metadata
    run_command(
        "head -10 output/metadata.csv",
        "Show first 10 rows of metadata.csv"
    )
    
    # Show image properties
    run_command(
        "file output/P001_3_05.png",
        "Check properties of sample image P001_3_05.png"
    )
    
    print("\n" + "="*60)
    print("✅ Workflow completed successfully!")
    print("="*60)
    print(f"\nOutput location: ./output/")
    print(f"  - 100 digit images (P001_0_01.png to P001_9_10.png)")
    print(f"  - metadata.csv with digit information")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
