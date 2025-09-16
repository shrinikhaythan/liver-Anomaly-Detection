import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_score(original, reconstructed):
    """
    Calculate an anomaly score between original and reconstructed images
    """
    try:
        # Calculate the absolute difference
        difference = np.abs(original - reconstructed)
        
        # Calculate mean difference as the anomaly score
        # You might want to adjust this based on your specific needs
        score = np.mean(difference)
        
        return float(score)
        
    except Exception as e:
        logger.error(f"Error calculating anomaly score: {e}")
        return 0.0

def get_traffic_light(score):
    """
    Convert anomaly score to traffic light flag
    Adjust these thresholds based on your model's performance
    """
    # These are example thresholds - you should adjust them based on your data
    if score < 0.02:
        return "Green"    # No or minimal anomaly
    elif score < 0.05:
        return "Yellow"   # Mild anomaly
    else:
        return "Red"      # Severe anomaly