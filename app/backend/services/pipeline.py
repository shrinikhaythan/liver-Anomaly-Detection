import os
import json
import numpy as np
from models import get_models
from models.cnn_model import segment_liver
from models.diffusion_model import generate_healthy_reconstruction
from utils.heatmap import create_heatmap
from utils.traffic_light import calculate_score, get_traffic_light
from utils.preprocessing import preprocess_ct_slice
import logging
import time

logger = logging.getLogger(__name__)

# Add a simple progress tracking mechanism
processing_progress = {}

def crop_with_mask(image, mask, threshold=0.5):
    """Crop image using the mask"""
    try:
        # Convert mask to binary
        binary_mask = mask > threshold
        
        # Find bounding box of the mask
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # If no mask found, return the whole image
            return image
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop the image
        cropped = image[rmin:rmax+1, cmin:cmax+1]
        
        return cropped
    except Exception as e:
        logger.error(f"Error in crop_with_mask: {e}")
        return image

def process_scan(scan_id, config):
    # Get the loaded models
    cnn_model, diffusion_model, device = get_models()
    
    if cnn_model is None or diffusion_model is None:
        raise ValueError("Models not loaded. Call load_models() first.")
    
    scan_path = os.path.join(config['UPLOAD_FOLDER'], scan_id)
    results_path = os.path.join(config['RESULTS_FOLDER'], scan_id)
    os.makedirs(results_path, exist_ok=True)
    
    # Get list of slices to process
    slice_files = [f for f in os.listdir(scan_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz', '.npy', '.npz'))]
    
    total_slices = len(slice_files)
    processing_progress[scan_id] = {
        'total': total_slices,
        'processed': 0,
        'start_time': time.time()
    }
    
    results = []
    
    for i, slice_file in enumerate(slice_files):
        slice_path = os.path.join(scan_path, slice_file)
        
        try:
            logger.info(f"Processing slice {i+1}/{total_slices}: {slice_file}")
            
            # Preprocessing
            preprocessed_image = preprocess_ct_slice(slice_path)
            if preprocessed_image is None:
                logger.warning(f"Skipping slice {slice_file}: preprocessing failed")
                continue
            
            # CNN Segmentation
            liver_mask = segment_liver(preprocessed_image, cnn_model)
            
            # Crop abdomen
            cropped_image = crop_with_mask(preprocessed_image, liver_mask)
            
            # Diffusion reconstruction
            healthy_reconstruction = generate_healthy_reconstruction(cropped_image, diffusion_model, device)
            
            # Calculate residual and heatmap
            heatmap_path = create_heatmap(cropped_image, healthy_reconstruction, results_path, slice_file)
            
            # Calculate score and flag
            score = calculate_score(cropped_image, healthy_reconstruction)
            flag = get_traffic_light(score)
            
            results.append({
                'sliceId': slice_file,
                'anomalyScore': float(score),
                'flag': flag,
                'heatmapPath': f'/results/{scan_id}/{os.path.basename(heatmap_path)}'
            })
            
            # Update progress
            processing_progress[scan_id]['processed'] = i + 1
            logger.info(f"Processed slice {slice_file}: score={score}, flag={flag}")
            
        except Exception as e:
            logger.error(f"Error processing slice {slice_file}: {e}")
            continue
    
    # Generate summary
    summary = generate_summary(results)
    
    # Save report
    report = {
        'patientId': scan_id,
        'totalSlices': len(results),
        'results': results,
        'summary': summary
    }
    
    with open(os.path.join(results_path, 'report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Calculate processing time
    processing_time = time.time() - processing_progress[scan_id]['start_time']
    logger.info(f"Completed processing scan {scan_id}: {len(results)} slices processed in {processing_time:.2f} seconds")
    
    # Remove from progress tracking
    if scan_id in processing_progress:
        del processing_progress[scan_id]
    
    return report

def generate_summary(results):
    # Count flags
    green_count = sum(1 for r in results if r['flag'] == 'Green')
    yellow_count = sum(1 for r in results if r['flag'] == 'Yellow')
    red_count = sum(1 for r in results if r['flag'] == 'Red')
    
    if red_count > 0:
        return f"Severe anomalies detected in {red_count} slices. Immediate review recommended."
    elif yellow_count > 0:
        return f"Mild anomalies detected in {yellow_count} slices. Recommend further review."
    else:
        return "No anomalies detected. All slices appear normal."