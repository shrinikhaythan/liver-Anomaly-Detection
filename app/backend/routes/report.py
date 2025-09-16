from flask import Blueprint, send_file, jsonify
import os
import json

report_bp = Blueprint('report', __name__)

@report_bp.route('/report/<scan_id>', methods=['GET'])
def get_report(scan_id):
    try:
        report_path = os.path.join(current_app.config['RESULTS_FOLDER'], scan_id, 'report.json')
        
        if not os.path.exists(report_path):
            return jsonify({'error': 'Report not found'}), 404
            
        with open(report_path, 'r') as f:
            report_data = json.load(f)
            
        return jsonify(report_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500