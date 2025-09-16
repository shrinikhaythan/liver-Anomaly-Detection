# Medical CT Analysis with AI Agents

## Real Implementation with Your Trained Model

This is the clean, optimized version using your actual trained diffusion model.

### What's Included:

#### Backend (Python Flask)
- ✅ **YOUR trained diffusion model** (`ddpm_ct_best_model.pt`)
- ✅ **Real processing pipeline** 
- ✅ **Real heatmap generation**
- ✅ **AI medical analysis** (with Gemini API)
- ✅ **Comprehensive medical reports**

#### Frontend (React TypeScript)
- ✅ **Modern medical dashboard**
- ✅ **AI analysis display**
- ✅ **Interactive slice viewer**
- ✅ **Traffic light alerts**
- ✅ **Doctor-friendly reports**

### Quick Start:

1. **Set your Gemini API key** (optional - works without it):
   ```cmd
   set GOOGLE_API_KEY=your_gemini_api_key_here
   ```

2. **Run the application**:
   ```cmd
   start.bat
   ```

3. **Access the app**:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:5000

### Features:

- 🔬 **Real Model Processing**: Uses your trained `ddpm_ct_best_model.pt`
- 🧠 **AI Medical Analysis**: Comprehensive medical reports
- 🎯 **Real Anomaly Detection**: Based on your model's outputs
- 🔥 **Real Heatmaps**: Generated from actual residuals
- 📊 **Traffic Light System**: Red/Yellow/Green based on real scores
- 📋 **Medical Reports**: Professional clinical analysis

### File Structure:
```
C:\temp\app\
├── backend\
│   ├── app.py          # Main backend with your model
│   └── model.pt        # Your trained diffusion model
├── frontend\
│   ├── src\            # React TypeScript source
│   ├── package.json    # Dependencies
│   └── ...             # Frontend files
├── start.bat           # Quick start script
└── README.md          # This file
```

### Notes:
- No dummy data - uses real uploaded files
- Works with/without CNN model (graceful fallback)
- AI analysis works with/without API key (fallback medical knowledge)
- Generates real heatmaps from your model's outputs
