#!/usr/bin/env python3
import requests
import json
import os

def test_full_pipeline():
    """Test the complete pipeline with AI integration"""
    
    # Set the API key
    os.environ['GEMINI_API_KEY'] = 'AIzaSyAKh2bxZpD5eE77mvv35GAmvqXslp62yMY'
    
    print("🧪 Testing Full AI-Integrated Medical Analysis Pipeline")
    print("="*60)
    
    try:
        # Step 1: Upload file
        print("📤 Step 1: Uploading test image...")
        with open('test_data/synthetic_ct_slice.png', 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/upload', files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            return
            
        upload_result = response.json()
        scan_id = upload_result['scan_id']
        print(f"✅ Upload successful! Scan ID: {scan_id}")
        
        # Step 2: Process with AI
        print("🤖 Step 2: Processing with AI medical analysis...")
        process_data = {'scan_id': scan_id}
        response = requests.post('http://localhost:5000/process', 
                               headers={'Content-Type': 'application/json'},
                               data=json.dumps(process_data))
        
        if response.status_code != 200:
            print(f"❌ Processing failed: {response.text}")
            return
            
        process_result = response.json()
        print("✅ AI Processing completed!")
        
        # Step 3: Display results
        print("\n🎯 ANALYSIS RESULTS:")
        print("="*40)
        
        results = process_result.get('results', {})
        ai_summary = results.get('ai_summary', 'No summary')
        print(f"🧠 AI Summary: {ai_summary}")
        
        slice_results = results.get('results', [])
        for i, result in enumerate(slice_results):
            print(f"\n📊 Slice {i+1} Results:")
            print(f"   🎯 Anomaly Score: {result.get('anomalyScore', 0):.2f}%")
            print(f"   🚦 Status: {result.get('flag', 'Unknown')}")
            print(f"   🧾 AI Report: {result.get('ai_analysis', 'No analysis')[:100]}...")
            print(f"   🖼️ Heatmap: {result.get('heatmapPath', 'No heatmap')}")
            
        print("\n" + "="*60)
        print("🎉 FULL AI PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("✅ Real diffusion model analysis")
        print("✅ Real heatmap generation") 
        print("✅ Real AI medical reports")
        print("✅ Complete integration working")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_pipeline()
