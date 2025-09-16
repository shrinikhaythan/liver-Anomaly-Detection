#!/usr/bin/env python3
import requests
import json

def test_upload():
    """Test uploading image to the running server"""
    
    try:
        print("🔬 Testing image upload and analysis...")
        
        # Upload the synthetic test image
        with open('test_data/synthetic_ct_slice.png', 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/api/upload', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Analysis completed:")
            print(json.dumps(result, indent=2))
            
            # Check if heatmap was generated
            if result.get('results') and len(result['results']) > 0:
                first_result = result['results'][0]
                heatmap_file = first_result.get('heatmap_file')
                if heatmap_file:
                    print(f"🎯 Heatmap generated: {heatmap_file}")
                    print(f"📊 Anomaly score: {first_result.get('anomaly_score', 'N/A')}%")
                    print(f"🚦 Status: {first_result.get('status', 'Unknown')}")
                    
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_upload()
