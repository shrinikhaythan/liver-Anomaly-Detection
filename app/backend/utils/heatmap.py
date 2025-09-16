import numpy as np
import cv2
import os
import logging
from matplotlib import cm

logger = logging.getLogger(__name__)

def create_heatmap(original, reconstructed, output_dir, filename):
    """
    Create a heatmap showing the difference between original and reconstructed images
    """
    try:
        # Calculate the absolute difference
        difference = np.abs(original - reconstructed)
        
        # Normalize the difference to [0, 1] for better visualization
        if np.max(difference) > 0:
            difference_normalized = difference / np.max(difference)
        else:
            difference_normalized = difference
        
        # Apply a colormap to the difference
        heatmap = cm.jet(difference_normalized)[:, :, :3]  # Get RGB, ignore alpha
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Convert original to 3-channel for blending if it's grayscale
        if len(original.shape) == 2:
            original_rgb = np.stack([original] * 3, axis=-1)
        else:
            original_rgb = original
        
        # Scale original to 0-255 if it's in 0-1 range
        if original_rgb.max() <= 1.0:
            original_rgb = (original_rgb * 255).astype(np.uint8)
        
        # Blend the heatmap with the original image
        alpha = 0.6
        blended = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)
        
        # Create the output filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the image
        cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        raise