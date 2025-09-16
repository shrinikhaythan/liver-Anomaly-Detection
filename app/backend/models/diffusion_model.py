import torch
from diffusers import DDIMScheduler
import gc
import logging
import numpy as np

logger = logging.getLogger(__name__)

def generate_healthy_reconstruction(cropped_image, model, device):
    """
    Use the diffusion model to generate a healthy reconstruction (CPU version)
    """
    try:
        # Convert numpy array to torch tensor if needed
        if isinstance(cropped_image, np.ndarray):
            cropped_image = torch.from_numpy(cropped_image).float()
        
        # Add batch and channel dimensions if needed
        if len(cropped_image.shape) == 2:
            cropped_image = cropped_image.unsqueeze(0).unsqueeze(0)
        elif len(cropped_image.shape) == 3:
            # Assume it's (H, W, C), convert to (C, H, W)
            cropped_image = cropped_image.permute(2, 0, 1).unsqueeze(0)
        
        # Move to device and scale to [-1, 1]
        cropped_image = cropped_image.to(device)
        cropped_image = cropped_image * 2.0 - 1.0  # Scale from [0,1] to [-1,1]
        
        # Setup DDIM scheduler
        ddim_scheduler = DDIMScheduler.from_config(model.config)
        ddim_scheduler.set_timesteps(num_inference_steps=50)
        
        # Generate healthy reconstruction
        generated_sample = cropped_image.clone()
        
        with torch.no_grad():
            for t in ddim_scheduler.timesteps:
                # CPU version doesn't use autocast
                noise_pred = model(generated_sample, t).sample
                generated_sample = ddim_scheduler.step(noise_pred, t, generated_sample).prev_sample
                
                # Clear memory to prevent crashes
                gc.collect()
        
        # Scale back to [0, 1]
        generated_sample = (generated_sample + 1) / 2.0
        
        # Convert to numpy and remove batch and channel dimensions
        if len(generated_sample.shape) == 4:
            generated_sample = generated_sample[0]  # Remove batch dimension
        
        if len(generated_sample.shape) == 3:
            generated_sample = generated_sample[0]  # Remove channel dimension if it's 1
        
        return generated_sample.cpu().numpy()
        
    except Exception as e:
        logger.error(f"Error in diffusion generation: {e}")
        raise