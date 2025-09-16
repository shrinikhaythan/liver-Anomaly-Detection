#!/usr/bin/env python3
"""
Simplified Medical Image Analysis Backend
Focus on working diffusion model with real data processing
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import zipfile
from diffusers import UNet2DModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# Global model storage
diffusion_model = None
model_loaded = False

def load_diffusion_model():
    """Load the trained diffusion model using diffusers UNet2DModel"""
    global diffusion_model, model_loaded
    
    try:
        model_path = "model.pt"
        if not os.path.exists(model_path):
            model_path = "trained_models/diffusion_model.pt"
            
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found at {model_path}")
            return False
            
        logger.info(f"Loading diffusion model from: {model_path}")
        
        # Initialize model architecture using diffusers (same as working version)
        diffusion_model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            attention_head_dim=8,
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load state dict
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        diffusion_model.eval()
        model_loaded = True
        
        logger.info("✅ Diffusion model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load diffusion model: {e}")
        logger.error(traceback.format_exc())
        return False

def preprocess_image(image_array):
    """Preprocess image for model input"""
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2)
    
    # Resize to model input size
    img = Image.fromarray(image_array).resize((256, 256))
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def generate_heatmap(original, reconstruction):
    """Generate anomaly heatmap from difference between original and reconstruction"""
    diff = np.abs(original.squeeze() - reconstruction.squeeze())
    
    # Normalize difference to [0, 1]
    if diff.max() > 0:
        diff = diff / diff.max()
    
    # Create heatmap
    heatmap = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    
    # Map to color: blue (normal) to red (anomalous)
    heatmap[:, :, 2] = (255 * (1 - diff)).astype(np.uint8)  # Blue channel
    heatmap[:, :, 0] = (255 * diff).astype(np.uint8)        # Red channel
    
    return heatmap

def analyze_image(image_array):
    """Analyze image using diffusion model"""
    if not model_loaded or diffusion_model is None:
        return None, None, "Model not loaded"
    
    try:
        # Preprocess
        input_tensor = preprocess_image(image_array)
        
        # Run inference
        with torch.no_grad():
            reconstruction = diffusion_model(input_tensor)
        
        # Convert to numpy
        original_np = input_tensor.squeeze().cpu().numpy()
        reconstruction_np = reconstruction.squeeze().cpu().numpy()
        
        # Generate heatmap
        heatmap = generate_heatmap(original_np, reconstruction_np)
        
        # Calculate anomaly score
        diff = np.abs(original_np - reconstruction_np)
        anomaly_score = float(np.mean(diff))
        
        return heatmap, anomaly_score, None
        
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        return None, None, str(e)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create upload directory
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        results = []
        
        # Handle different file types
        filename = file.filename.lower()
        
        if filename.endswith('.zip'):
            # Extract and process ZIP file
            with zipfile.ZipFile(file.stream, 'r') as zip_ref:
                for file_info in zip_ref.filelist[:3]:  # Limit to 3 files for demo
                    if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with zip_ref.open(file_info) as img_file:
                            image = Image.open(img_file).convert('L')
                            image_array = np.array(image)
                            
                            # Analyze image
                            heatmap, anomaly_score, error = analyze_image(image_array)
                            
                            if error:
                                results.append({
                                    'filename': file_info.filename,
                                    'error': error
                                })
                            else:
                                results.append({
                                    'filename': file_info.filename,
                                    'anomaly_score': anomaly_score,
                                    'status': 'normal' if anomaly_score < 0.1 else 'suspicious' if anomaly_score < 0.2 else 'anomalous'
                                })
        
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            # Process single image
            image = Image.open(file.stream).convert('L')
            image_array = np.array(image)
            
            # Analyze image
            heatmap, anomaly_score, error = analyze_image(image_array)
            
            if error:
                return jsonify({'error': error}), 500
            
            results.append({
                'filename': file.filename,
                'anomaly_score': anomaly_score,
                'status': 'normal' if anomaly_score < 0.1 else 'suspicious' if anomaly_score < 0.2 else 'anomalous'
            })
        
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        return jsonify({
            'message': 'Analysis completed',
            'results': results,
            'model_info': {
                'diffusion_model': 'UNet2D',
                'analysis_type': 'Anomaly Detection via Reconstruction'
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models
    logger.info("🚀 Starting Medical Image Analysis Backend...")
    
    if load_diffusion_model():
        logger.info("✅ Ready for real data analysis!")
    else:
        logger.warning("⚠️ Models not loaded - running in error mode")
    
    logger.info("📍 Backend running on: http://localhost:5000")
    logger.info("📊 Health check available at: http://localhost:5000/api/health")
    logger.info("📤 Upload endpoint: http://localhost:5000/api/upload")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
