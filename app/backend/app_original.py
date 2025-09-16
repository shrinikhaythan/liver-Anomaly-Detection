from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from routes.upload import upload_bp
from routes.process import process_bp
from routes.report import report_bp
from routes.progress import progress_bp
from models import load_models
from utils.optimization import optimize_for_cpu
import logging
import os

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
    app.config['RESULTS_FOLDER'] = 'results'
    
    # Set model paths (relative to backend directory)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    app.config['CNN_MODEL_PATH'] = os.environ.get('CNN_MODEL_PATH', os.path.join(backend_dir, "trained Model", "liver_unet.h5"))
    app.config['DIFFUSION_MODEL_PATH'] = os.environ.get('DIFFUSION_MODEL_PATH', os.path.join(backend_dir, "trained Model", "ddpm_ct_best_model.pt"))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Optimize for CPU
    optimize_for_cpu()
    
    # Load models on app startup (force CPU)
    if load_models(app.config['CNN_MODEL_PATH'], app.config['DIFFUSION_MODEL_PATH']):
        logger.info("All models loaded successfully for CPU")
    else:
        logger.error("Failed to load models. The application may not work correctly.")
    
    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(process_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(progress_bp)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        from models import models_loaded
        return jsonify({
            'status': 'healthy',
            'models_loaded': models_loaded()
        }), 200
    
    # Serve static files from results folder
    @app.route('/results/<path:filename>')
    def serve_results(filename):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)