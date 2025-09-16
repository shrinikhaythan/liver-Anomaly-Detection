from flask import Blueprint, request, jsonify, current_app
from services.pipeline import process_scan

process_bp = Blueprint('process', __name__)

@process_bp.route('/process', methods=['POST'])
def process_scan_route():
    try:
        data = request.get_json()
        scan_id = data.get('scan_id')
        
        if not scan_id:
            return jsonify({'error': 'scan_id required'}), 400
        
        results = process_scan(scan_id, current_app.config)
        
        return jsonify({
            'message': 'Processing completed',
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500