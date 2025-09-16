#!/usr/bin/env python3
"""
Inspect the actual model structure from the checkpoint
"""

import torch
import pprint

def inspect_checkpoint():
    """Load and inspect the checkpoint structure"""
    
    try:
        checkpoint = torch.load("model.pt", map_location='cpu')
        print("=== CHECKPOINT STRUCTURE ===")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n=== MODEL STATE DICT KEYS ===")
            print(f"Total parameters: {len(model_state)}")
            
            # Group parameters by layer
            layers = {}
            for key in model_state.keys():
                layer_name = key.split('.')[0]  # Get first part before dot
                if layer_name not in layers:
                    layers[layer_name] = []
                layers[layer_name].append(key)
            
            print(f"\n=== LAYER STRUCTURE ===")
            for layer, params in layers.items():
                print(f"\n{layer}:")
                for param in sorted(params):
                    shape = model_state[param].shape
                    print(f"  {param}: {shape}")
            
            # Try to infer architecture
            print(f"\n=== ARCHITECTURE INFERENCE ===")
            
            # Look for common patterns
            conv_layers = [k for k in model_state.keys() if 'conv' in k and 'weight' in k]
            bn_layers = [k for k in model_state.keys() if ('bn' in k or 'norm' in k) and 'weight' in k]
            
            print(f"Convolution layers found: {len(conv_layers)}")
            print(f"BatchNorm/GroupNorm layers found: {len(bn_layers)}")
            
            # Print first few conv layers to understand structure
            print(f"\nFirst 10 convolution layers:")
            for conv in sorted(conv_layers)[:10]:
                shape = model_state[conv].shape
                print(f"  {conv}: {shape}")
                
        else:
            print("No 'model_state_dict' found in checkpoint!")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_checkpoint()
