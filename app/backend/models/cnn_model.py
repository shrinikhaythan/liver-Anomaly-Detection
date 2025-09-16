import numpy as np
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

def segment_liver(image, model):
    """
    Use the CNN model to segment the liver from the preprocessed image
    """
    try:
        # Prepare the image for the model
        if len(image.shape) == 2:
            # Add channel and batch dimensions
            input_image = np.expand_dims(image, axis=-1)
            input_image = np.expand_dims(input_image, axis=0)
        else:
            # Already has channel dimension, just add batch dimension
            input_image = np.expand_dims(image, axis=0)
        
        # Predict mask using the CNN model
        mask = model.predict(input_image, verbose=0)
        
        # Remove batch dimension
        if len(mask.shape) == 4:
            mask = mask[0]
        
        # If mask has multiple channels, take the first one
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
            
        return mask
        
    except Exception as e:
        logger.error(f"Error in liver segmentation: {e}")
        raise

def load_cnn_model(model_path):
    """
    Load the CNN model from the specified path
    """
    try:
        import tensorflow as tf
        # Disable GPU for TensorFlow to avoid conflicts with PyTorch
        tf.config.set_visible_devices([], 'GPU')
        
        model = tf.keras.models.load_model(model_path)
        logger.info(f"CNN model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading CNN model: {e}")
        return None