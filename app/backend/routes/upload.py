from flask import Blueprint, request, jsonify, current_app
from utils.file_handling import extract_zip, count_slices
import os

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save and extract zip
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)
        extract_path = extract_zip(upload_path)
        
        # Count slices
        slice_count = count_slices(extract_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'slice_count': slice_count,
            'scan_id': os.path.basename(extract_path)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500