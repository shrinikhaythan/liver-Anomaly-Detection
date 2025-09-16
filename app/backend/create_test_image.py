#!/usr/bin/env python3
"""
Create synthetic CT test image for testing the medical analysis pipeline
"""

import numpy as np
from PIL import Image
import os

def create_synthetic_ct_image():
    """Create a synthetic CT-like image with some anomalies"""
    
    # Create 512x512 image with grayscale values typical of CT scans
    img_size = 512
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Add background tissue (soft tissue values ~50-100 HU -> grayscale ~100-150)
    img.fill(120)
    
    # Add some circular structures (organs)
    center_x, center_y = img_size // 2, img_size // 2
    
    # Large organ (liver-like)
    for y in range(img_size):
        for x in range(img_size):
            dist = np.sqrt((x - center_x - 50)**2 + (y - center_y)**2)
            if dist < 80:
                img[y, x] = 140
    
    # Smaller organ (kidney-like)
    for y in range(img_size):
        for x in range(img_size):
            dist = np.sqrt((x - center_x + 100)**2 + (y - center_y - 50)**2)
            if dist < 40:
                img[y, x] = 160
    
    # Add some "anomalies" (bright spots that could be lesions)
    # Anomaly 1
    for y in range(center_y - 20, center_y + 20):
        for x in range(center_x - 20, center_x + 20):
            if 0 <= y < img_size and 0 <= x < img_size:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 15:
                    img[y, x] = 220  # Bright spot
    
    # Anomaly 2
    for y in range(center_y + 50, center_y + 80):
        for x in range(center_x - 80, center_x - 50):
            if 0 <= y < img_size and 0 <= x < img_size:
                dist = np.sqrt((x - (center_x - 65))**2 + (y - (center_y + 65))**2)
                if dist < 12:
                    img[y, x] = 200  # Another bright spot
    
    # Add some noise for realism
    noise = np.random.normal(0, 5, (img_size, img_size))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def main():
    """Create test images"""
    
    # Create test_data directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Create synthetic CT image
    ct_image = create_synthetic_ct_image()
    
    # Save as PNG
    img_pil = Image.fromarray(ct_image, mode='L')
    img_pil.save('test_data/synthetic_ct_slice.png')
    
    print("✅ Created synthetic CT test image: test_data/synthetic_ct_slice.png")
    print(f"   Image size: {ct_image.shape}")
    print(f"   Value range: {ct_image.min()} - {ct_image.max()}")
    print("   Contains simulated anomalies for testing")

if __name__ == "__main__":
    main()
