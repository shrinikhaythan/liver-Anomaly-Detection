from flask import Blueprint, jsonify
from services.pipeline import processing_progress

progress_bp = Blueprint('progress', __name__)

@progress_bp.route('/progress/<scan_id>', methods=['GET'])
def get_progress(scan_id):
    try:
        if scan_id in processing_progress:
            progress_data = processing_progress[scan_id]
            return jsonify({
                'total': progress_data['total'],
                'processed': progress_data['processed'],
                'percentage': round((progress_data['processed'] / progress_data['total']) * 100, 2) if progress_data['total'] > 0 else 0
            }), 200
        else:
            return jsonify({
                'total': 0,
                'processed': 0,
                'percentage': 0,
                'message': 'No processing in progress for this scan'
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
