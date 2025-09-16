#!/usr/bin/env python3
"""
Simple test to load the diffusion model
"""

import torch
from diffusers import UNet2DModel, DDIMScheduler
import os

def test_model_loading():
    try:
        print("🔄 Testing diffusion model loading...")
        
        device = "cpu"
        model_path = r"C:\temp\models\diffusion.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at {model_path}")
            return False
        
        print(f"📁 Found model file: {model_path}")
        print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        # Define model architecture
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
        
        print("✅ Model architecture created")
        
        # Load weights
        print("🔄 Loading model weights...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        diffusion_model.to(device)
        diffusion_model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"🎯 Model parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
        
        # Test inference
        print("🔄 Testing inference...")
        test_input = torch.randn(1, 1, 256, 256).to(device)
        timestep = torch.tensor([10]).to(device)
        
        with torch.no_grad():
            output = diffusion_model(test_input, timestep).sample
        
        print(f"✅ Inference successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print(f"\n{'🎉 SUCCESS!' if success else '❌ FAILED!'}")
