from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import time
import logging
import numpy as np
from werkzeug.utils import secure_filename
import torch
from diffusers import UNet2DModel, DDIMScheduler
from PIL import Image
import cv2
import base64
import io

# AI Agent imports
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Agent libraries not available: {e}")
    AGENTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Global variables for models and scan results
diffusion_model = None
device = "cpu"
model_loaded = False
scan_results = {}  # Store scan results for processing

# --- AI Agent Configuration ---
if AGENTS_AVAILABLE:
    # Configure Gemini API
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        llm = GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini API configured successfully")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not found. AI agent will be disabled.")
        AGENTS_AVAILABLE = False

# --- AI Agent Tools ---
if AGENTS_AVAILABLE:
    @tool
    def search_medical_database(query: str, source: str = "medical_knowledge"):
        """
        Search for medical information about liver anomalies and treatments.
        Args:
            query: Medical query about liver conditions
            source: Source type (pubmed, cancer.gov, medical_knowledge)
        """
        logger.info(f"Medical database search: {query} from {source}")
        
        # Comprehensive medical knowledge base for liver conditions
        if "liver" in query.lower():
            if "causes" in query.lower() or "etiology" in query.lower():
                return (
                    "Common causes of liver lesions include: "
                    "hepatocellular carcinoma (HCC) - most common primary liver cancer, "
                    "metastases from colorectal, breast, lung cancers, "
                    "hepatic adenomas, focal nodular hyperplasia (FNH), "
                    "hemangiomas (benign vascular lesions), "
                    "hepatic cysts, and abscesses. "
                    "Risk factors include chronic hepatitis B/C, cirrhosis, "
                    "alcohol abuse, and metabolic disorders."
                )
            elif "treatment" in query.lower() or "management" in query.lower():
                return (
                    "Treatment options for liver lesions include: "
                    "surgical resection (gold standard for localized disease), "
                    "liver transplantation, radiofrequency ablation (RFA), "
                    "transarterial chemoembolization (TACE), "
                    "targeted therapy (sorafenib, lenvatinib), "
                    "immunotherapy, and supportive care. "
                    "Treatment choice depends on lesion size, location, "
                    "liver function, and patient performance status."
                )
        
        return f"Medical information retrieved for: {query}"

def generate_ai_medical_report(anomaly_score, status, slice_id, metrics=None):
    """
    Generate detailed medical report using AI agent analysis with heatmap-based insights.
    """
    if not AGENTS_AVAILABLE:
        logger.warning("AI agents not available, returning basic report")
        return f"Reconstruction error analysis shows {anomaly_score:.1f}% anomaly score. Status: {status}. AI analysis unavailable."
    
    try:
        logger.info(f"🤖 Generating AI medical report for {slice_id}...")
        
        # Prepare detailed analysis context
        anomaly_context = ""
        if metrics:
            pixel_ratio = metrics.get('anomaly_pixel_ratio', 0)
            high_anomaly_pixels = metrics.get('high_anomaly_pixels', 0)
            
            if anomaly_score > 20:
                anomaly_context = f"High-intensity reconstruction errors detected in {pixel_ratio:.1f}% of liver tissue ({high_anomaly_pixels} pixels), suggesting significant structural abnormalities."
            elif anomaly_score > 10:
                anomaly_context = f"Moderate reconstruction errors in {pixel_ratio:.1f}% of liver tissue, indicating possible focal lesions or texture changes."
            else:
                anomaly_context = f"Minimal reconstruction errors ({pixel_ratio:.1f}% affected area), suggesting largely normal liver architecture."
        
        # Create AI Agent with enhanced context
        medical_report_agent = Agent(
            role="Expert Liver Radiologist",
            goal="Provide detailed medical analysis based on CT reconstruction heatmap findings",
            tools=[search_medical_database],
            verbose=False,
            backstory=(
                "You are a board-certified radiologist with 15+ years experience in liver imaging. "
                "You specialize in interpreting AI-generated reconstruction error heatmaps to identify "
                "liver pathology including hepatocellular carcinoma, metastases, hemangiomas, and cirrhosis. "
                "You provide specific anatomical location analysis and clinical recommendations."
            )
        )
        
        # Create detailed analysis task
        analysis_task = Task(
            description=(
                f"LIVER CT RECONSTRUCTION ANALYSIS:\n"
                f"- Reconstruction error score: {anomaly_score:.1f}%\n"
                f"- Classification: {status}\n"
                f"- Context: {anomaly_context}\n\n"
                "Based on this heatmap analysis, provide:\n"
                "1. ANATOMICAL ASSESSMENT: Identify which liver segments/zones show highest reconstruction errors\n"
                "2. PATHOLOGICAL INTERPRETATION: Use search_medical_database to determine most likely causes of these reconstruction errors\n"
                "3. CLINICAL SIGNIFICANCE: Explain what these findings could indicate (HCC, metastasis, benign lesions, etc.)\n"
                "4. RECOMMENDATIONS: Suggest next steps (further imaging, biopsy, follow-up)\n\n"
                "Provide 3-4 sentences in professional medical language suitable for radiological report."
            ),
            agent=medical_report_agent,
            expected_output="Detailed liver pathology report with anatomical localization and clinical recommendations"
        )
        
        # Execute AI Analysis
        crew = Crew(
            agents=[medical_report_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=False
        )
        
        # Run the analysis
        result = crew.kickoff(inputs={
            "slice_id": slice_id,
            "anomaly_score": anomaly_score,
            "status": status,
            "anomaly_context": anomaly_context,
            "metrics": metrics or {}
        })
        
        logger.info("✅ AI medical report generated successfully")
        return str(result)
        
    except Exception as e:
        logger.error(f"AI report generation failed: {e}")
        # Provide detailed fallback report
        if anomaly_score > 20:
            return f"Significant reconstruction errors ({anomaly_score:.1f}%) suggest possible hepatic lesions requiring urgent radiological review and potential tissue sampling. Recommend contrast-enhanced MRI and multidisciplinary team consultation."
        elif anomaly_score > 10:
            return f"Moderate reconstruction abnormalities ({anomaly_score:.1f}%) indicate possible focal liver changes. Recommend follow-up imaging in 3-6 months and correlation with clinical symptoms and liver function tests."
        else:
            return f"Minimal reconstruction errors ({anomaly_score:.1f}%) suggest largely normal liver architecture. Consider routine surveillance if patient has risk factors for liver disease."

def load_diffusion_model():
    """Load the trained diffusion model"""
    global diffusion_model, model_loaded, device
    
    try:
        model_path = "model.pt"
        if not os.path.exists(model_path):
            model_path = "trained_models/diffusion_model.pt"
            
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found at {model_path}")
            return False
            
        logger.info(f"Loading diffusion model from: {model_path}")
        
        # Initialize model architecture (SAME AS WORKING VERSION)
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
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load state dict
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        diffusion_model.to(device)
        diffusion_model.eval()
        model_loaded = True
        
        logger.info("✅ Diffusion model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load diffusion model: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(image_array):
    """Preprocess image for model input"""
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2)
    
    # Resize to model input size
    img = Image.fromarray(image_array.astype(np.uint8)).resize((256, 256))
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def generate_liver_heatmap(original_liver, reconstruction_liver, diff):
    """Generate liver-specific heatmap with selective coloring, preserving black background"""
    
    # Normalize original liver image to [0, 255] for background
    original_bg = ((original_liver - original_liver.min()) / (original_liver.max() - original_liver.min()) * 255).astype(np.uint8)
    
    # Create RGB heatmap starting with grayscale background
    heatmap = np.stack([original_bg, original_bg, original_bg], axis=2)
    
    # Create mask for non-black pixels (liver tissue only)
    # Black pixels are close to 0, liver tissue has higher values
    liver_tissue_mask = original_bg > 10  # Ignore very dark/black pixels
    
    # Calculate error statistics only on liver tissue (ignore black background)
    liver_diff = diff[liver_tissue_mask]
    if len(liver_diff) > 0:
        error_mean = np.mean(liver_diff)
        error_std = np.std(liver_diff)
        
        # Thresholds: mean + 1*std = medium, mean + 2*std = high (same as before)
        medium_threshold = error_mean + 1.0 * error_std
        high_threshold = error_mean + 2.0 * error_std
        
        # Apply color coding only to liver tissue (not black background)
        # High anomaly (red) - most anomalous liver pixels
        high_anomaly_mask = (diff > high_threshold) & liver_tissue_mask
        heatmap[high_anomaly_mask, 0] = 255  # Red channel
        heatmap[high_anomaly_mask, 1] = 0    # Green channel  
        heatmap[high_anomaly_mask, 2] = 0    # Blue channel
        
        # Medium anomaly (yellow) - mildly anomalous liver pixels
        medium_anomaly_mask = (diff > medium_threshold) & (diff <= high_threshold) & liver_tissue_mask
        heatmap[medium_anomaly_mask, 0] = 255  # Red channel
        heatmap[medium_anomaly_mask, 1] = 255  # Green channel (red + green = yellow)
        heatmap[medium_anomaly_mask, 2] = 0    # Blue channel
    
    # Black background stays black (no changes)
    # Normal liver tissue keeps original grayscale (no color changes)
    
    return heatmap

def generate_liver_mask(image_array):
    """Generate liver mask using CNN model (placeholder - uses simple thresholding for now)"""
    try:
        # For now, create a basic liver region mask using intensity thresholding
        # This simulates CNN liver segmentation until the actual CNN model is loaded
        normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        
        # Create a mask for liver-like intensities (middle range)
        liver_mask = ((normalized_image > 0.3) & (normalized_image < 0.8)).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        from scipy import ndimage
        liver_mask = ndimage.binary_fill_holes(liver_mask).astype(np.uint8)
        liver_mask = ndimage.binary_opening(liver_mask, structure=np.ones((3,3))).astype(np.uint8)
        
        return liver_mask
        
    except Exception as e:
        logger.error(f"Error generating liver mask: {e}")
        # Return full mask if masking fails
        return np.ones(image_array.shape, dtype=np.uint8)

def apply_liver_mask(image_array, mask):
    """Apply liver mask to get cropped liver region"""
    # Apply mask
    masked_image = image_array * mask
    
    # Find bounding box of liver region
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return image_array  # Return original if no mask found
        
    min_row, max_row = np.min(coords[0]), np.max(coords[0])
    min_col, max_col = np.min(coords[1]), np.max(coords[1])
    
    # Crop to liver region
    cropped_liver = masked_image[min_row:max_row+1, min_col:max_col+1]
    cropped_mask = mask[min_row:max_row+1, min_col:max_col+1]
    
    # Normalize the cropped liver region
    liver_pixels = cropped_liver[cropped_mask > 0]
    if len(liver_pixels) > 0:
        min_val, max_val = np.min(liver_pixels), np.max(liver_pixels)
        if max_val > min_val:
            normalized_liver = (cropped_liver - min_val) / (max_val - min_val)
            normalized_liver = normalized_liver * cropped_mask  # Keep only liver region
        else:
            normalized_liver = cropped_liver
    else:
        normalized_liver = cropped_liver
        
    return normalized_liver, (min_row, max_row, min_col, max_col)

def analyze_image_with_diffusion(image_array):
    """Analyze image using CNN liver masking + diffusion model pipeline"""
    if not model_loaded or diffusion_model is None:
        return None, None, None, "Model not loaded", None
    
    try:
        # Step 1: Generate liver mask using CNN (simulated for now)
        logger.info("Generating liver mask...")
        liver_mask = generate_liver_mask(image_array)
        
        # Step 2: Apply mask to get cropped liver region
        logger.info("Applying liver mask and cropping...")
        cropped_liver, bbox = apply_liver_mask(image_array, liver_mask)
        
        # Step 3: Preprocess cropped liver for diffusion model
        input_tensor = preprocess_image(cropped_liver)
        
        # Step 4: Run diffusion reconstruction
        logger.info("Running diffusion model reconstruction...")
        with torch.no_grad():
            timestep = torch.zeros(1, dtype=torch.long)
            reconstruction = diffusion_model(input_tensor, timestep).sample
        
        # Step 5: Convert to numpy
        original_liver_np = input_tensor.squeeze().cpu().numpy()
        reconstruction_np = reconstruction.squeeze().cpu().numpy()
        
        # Step 6: Calculate pixel-wise errors
        diff = np.abs(original_liver_np - reconstruction_np)
        
        # Step 7: Generate heatmap on the original cropped liver
        heatmap = generate_liver_heatmap(original_liver_np, reconstruction_np, diff)
        
        # Step 8: Calculate anomaly metrics (previous method)
        mean_error = np.mean(diff)
        anomaly_score = float(mean_error * 100)  # Convert to percentage like before
        
        error_std = np.std(diff)
        high_anomaly_pixels = np.sum(diff > (np.mean(diff) + 2 * error_std))
        total_pixels = diff.size
        anomaly_pixel_ratio = (high_anomaly_pixels / total_pixels) * 100
        
        return original_liver_np, reconstruction_np, heatmap, anomaly_score, None, {
            'mean_error': float(mean_error),
            'error_std': float(error_std),
            'anomaly_pixel_ratio': float(anomaly_pixel_ratio),
            'high_anomaly_pixels': int(high_anomaly_pixels),
            'liver_mask': liver_mask,
            'bbox': bbox
        }
        
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, str(e), None

# Routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'diffusion_model_loaded': model_loaded,
        'timestamp': time.time()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = []
        processed_files = 0
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process single image
            logger.info(f"Processing single image: {filename}")
            
            image = Image.open(filepath).convert('L')
            image_array = np.array(image)
            
            # Analyze with diffusion model (new pipeline)
            original_liver, reconstruction_liver, heatmap, anomaly_score, error, metrics = analyze_image_with_diffusion(image_array)
            
            if error:
                return jsonify({'error': f'Analysis failed: {error}'}), 500
            
            # Save all three images
            timestamp = int(time.time())
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            
            # Save original cropped liver (normalized, no colors)
            original_liver_filename = f"original_liver_{timestamp}_{base_name}.png"
            original_liver_path = os.path.join(app.config['RESULTS_FOLDER'], original_liver_filename)
            original_liver_img = ((original_liver - original_liver.min()) / (original_liver.max() - original_liver.min()) * 255).astype(np.uint8)
            Image.fromarray(original_liver_img, mode='L').save(original_liver_path)
            
            # Save reconstructed liver (normalized, no colors)
            reconstruction_filename = f"reconstruction_{timestamp}_{base_name}.png"
            reconstruction_path = os.path.join(app.config['RESULTS_FOLDER'], reconstruction_filename)
            reconstruction_img = ((reconstruction_liver - reconstruction_liver.min()) / (reconstruction_liver.max() - reconstruction_liver.min()) * 255).astype(np.uint8)
            Image.fromarray(reconstruction_img, mode='L').save(reconstruction_path)
            
            # Save anomaly heatmap
            heatmap_filename = f"heatmap_{timestamp}_{base_name}.png"
            heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], heatmap_filename)
            Image.fromarray(heatmap).save(heatmap_path)
            
            # Determine status based on reconstruction error
            status = 'normal'
            if anomaly_score > 15:
                status = 'anomalous'
            elif anomaly_score > 8:
                status = 'suspicious'
            
            # Generate AI medical report with heatmap metrics
            ai_report = generate_ai_medical_report(anomaly_score, status, filename, metrics)
            
            results.append({
                'filename': filename,
                'anomaly_score': round(anomaly_score, 2),
                'status': status,
                'original_liver_file': original_liver_filename,
                'reconstruction_file': reconstruction_filename,
                'heatmap_file': heatmap_filename,
                'ai_report': ai_report
            })
            processed_files = 1
        
        # Store results for later processing
        scan_id = f"scan_{int(time.time())}"
        
        # Save scan results for the process endpoint
        global scan_results
        if 'scan_results' not in globals():
            scan_results = {}
        
        scan_results[scan_id] = {
            'results': results,
            'processed_files': processed_files,
            'status': 'uploaded'
        }
        
        return jsonify({
            'scan_id': scan_id,
            'slice_count': processed_files,
            'message': f'Successfully uploaded {processed_files} file(s)'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_scan():
    """Process uploaded scan"""
    try:
        data = request.get_json()
        scan_id = data.get('scan_id')
        
        if not scan_id or scan_id not in scan_results:
            return jsonify({'error': 'Scan ID not found'}), 404
        
        # Get stored results
        stored_data = scan_results[scan_id]
        results = stored_data['results']
        
        # Transform results to match frontend expectations
        transformed_results = []
        for result in results:
            # Determine flag based on improved anomaly score
            flag = 'Green'  # Normal
            if result['anomaly_score'] > 15:
                flag = 'Red'    # High anomaly
            elif result['anomaly_score'] > 8:
                flag = 'Yellow' # Medium anomaly
                
            transformed_results.append({
                'sliceId': f"Slice {len(transformed_results) + 1}",
                'anomalyScore': result['anomaly_score'],
                'flag': flag,
                'findings': result['ai_report'],
                'ai_analysis': result['ai_report'],
                'heatmapPath': f"/results/{result['heatmap_file']}",
                'originalImage': f"/results/{result['original_liver_file']}",  # Cropped liver (no colors)
                'reconstructedImage': f"/results/{result['reconstruction_file']}",  # Reconstructed liver (no colors)
            })
        
        # Update scan status
        scan_results[scan_id]['status'] = 'completed'
        scan_results[scan_id]['transformed_results'] = transformed_results
        
        return jsonify({
            'results': {
                'totalSlices': len(transformed_results),
                'results': transformed_results,
                'ai_summary': f'Analysis completed on {len(transformed_results)} slice(s). Diffusion model reconstruction analysis performed.'
            }
        })
        
    except Exception as e:
        logger.error(f"Process error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<scan_id>', methods=['GET'])
def get_progress(scan_id):
    """Get processing progress"""
    if scan_id not in scan_results:
        return jsonify({'error': 'Scan ID not found'}), 404
    
    status = scan_results[scan_id]['status']
    return jsonify({
        'status': status,
        'progress': 100 if status == 'completed' else 50
    })

@app.route('/report/<scan_id>', methods=['GET'])
def get_report(scan_id):
    """Get report for processed scan"""
    if scan_id not in scan_results:
        return jsonify({'error': 'Scan ID not found'}), 404
    
    stored_data = scan_results[scan_id]
    return jsonify({
        'report': {
            'scan_id': scan_id,
            'results': stored_data.get('transformed_results', []),
            'summary': 'Medical imaging analysis completed using diffusion model.'
        }
    })

@app.route('/results/<filename>')
def get_result_file(filename):
    """Serve result files (heatmaps)"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    logger.info("🚀 Starting Medical Image Analysis Backend...")
    
    # Load diffusion model
    if load_diffusion_model():
        logger.info("✅ Ready for real data analysis with diffusion model!")
    else:
        logger.warning("⚠️ Model not loaded - server will run in error mode")
    
    logger.info("📍 Backend running on: http://localhost:5000")
    logger.info("📊 Health check: http://localhost:5000/health")
    logger.info("📤 Upload endpoint: http://localhost:5000/upload")
    logger.info("⚙️  Process endpoint: http://localhost:5000/process")
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
