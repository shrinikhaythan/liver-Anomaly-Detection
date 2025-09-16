"""
Medical Image Analysis Backend
Integrated Flask application with real CNN and diffusion models
"""

import os
import sys
import logging
from flask import Flask
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create Flask application with proper configuration and model loading"""
    app = Flask(__name__)
    
    # Configure CORS for all routes
    CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])
    
    # Set Flask configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max file size
        UPLOAD_FOLDER=os.path.join(os.path.dirname(__file__), 'uploads'),
        RESULTS_FOLDER=os.path.join(os.path.dirname(__file__), 'results')
    )
    
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Initialize models when app starts
    with app.app_context():
        try:
            logger.info("Initializing models...")
            from models import initialize_models, get_models_status
            initialize_models()
            
            # Check model status
            status = get_models_status()
            if status['cnn_loaded'] and status['diffusion_loaded']:
                logger.info("✅ All models loaded successfully!")
            else:
                logger.warning("⚠️ Some models failed to load - check TensorFlow installation")
                logger.info(f"CNN Model: {'✅' if status['cnn_loaded'] else '❌'}")
                logger.info(f"Diffusion Model: {'✅' if status['diffusion_loaded'] else '❌'}")
                
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            logger.warning("Backend will run in fallback mode with dummy data")
    
    # Register blueprints
    try:
        from routes.health import health_bp
        from routes.upload import upload_bp
        from routes.analysis import analysis_bp
        
        app.register_blueprint(health_bp)
        app.register_blueprint(upload_bp)
        app.register_blueprint(analysis_bp)
        
        logger.info("✅ All routes registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to register routes: {e}")
        raise
    
    # Add error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(413)
    def too_large(error):
        return {'error': 'File too large. Maximum size is 100MB.'}, 413
    
    return app

if __name__ == '__main__':
    # Check if running from correct directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure we're in the backend directory
    if not os.path.exists(os.path.join(current_dir, 'models')):
        logger.error("❌ Models directory not found. Please run from backend directory.")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(current_dir, 'trained_models')):
        logger.error("❌ trained_models directory not found. Please ensure models are present.")
        sys.exit(1)
    
    # Create and run the application
    app = create_app()
    
    logger.info("🚀 Starting Medical Image Analysis Backend...")
    logger.info("📍 Backend running on: http://localhost:5000")
    logger.info("🔧 Frontend should connect to: http://localhost:5000/api")
    logger.info("📊 Health check available at: http://localhost:5000/api/health")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
