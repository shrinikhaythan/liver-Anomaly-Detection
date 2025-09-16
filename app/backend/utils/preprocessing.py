import os
import cv2
import pydicom
import numpy as np
from PIL import Image
from skimage.transform import resize

# Constants from your original code
TARGET_SIZE = (256, 256)
WINDOW_MIN, WINDOW_MAX = -200, 250

def read_dicom_array(path):
    """Read DICOM file and apply rescale slope/intercept"""
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        return arr * slope + intercept
    except Exception as e:
        raise ValueError(f"Error reading DICOM file {path}: {e}")

def window_and_normalize(img_hu, wmin=WINDOW_MIN, wmax=WINDOW_MAX):
    """Apply windowing and normalize to [0, 1] range"""
    img = np.clip(img_hu, wmin, wmax)
    img = (img - wmin) / (wmax - wmin)  # Normalize to 0-1
    return img.astype(np.float32)

def preprocess_ct_slice(image_path, target_shape=TARGET_SIZE):
    """
    Preprocess a CT slice from various formats (DICOM, PNG, JPG, etc.)
    Returns a normalized numpy array ready for model input
    """
    try:
        # Handle different file formats
        if image_path.lower().endswith('.dcm'):
            # DICOM file
            hu_arr = read_dicom_array(image_path)
            img_norm = window_and_normalize(hu_arr)
        elif image_path.lower().endswith(('.npy', '.npz')):
            # Numpy array file
            if image_path.endswith('.npz'):
                data = np.load(image_path)
                img_norm = data['image'] if 'image' in data else data['arr_0']
            else:
                img_norm = np.load(image_path)
        else:
            # Image file (PNG, JPG, etc.)
            img = np.array(Image.open(image_path).convert('L'))
            img_norm = img.astype(np.float32) / 255.0  # Normalize to 0-1
        
        # Resize if needed
        if img_norm.shape != target_shape:
            img_norm = resize(img_norm, target_shape, preserve_range=True)
        
        return img_norm
    except Exception as e:
        raise ValueError(f"Error preprocessing image {image_path}: {e}")