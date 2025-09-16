import os
import psutil
import logging

logger = logging.getLogger(__name__)

def optimize_for_cpu():
    """
    Optimize the environment for CPU processing
    """
    try:
        # Set environment variables for better CPU performance
        os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count(logical=False))
        os.environ['MKL_NUM_THREADS'] = str(psutil.cpu_count(logical=False))
        
        # For TensorFlow
        os.environ['TF_NUM_INTEROP_THREADS'] = str(psutil.cpu_count(logical=False))
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(psutil.cpu_count(logical=False))
        
        logger.info(f"Optimized for CPU with {psutil.cpu_count(logical=False)} physical cores")
    except Exception as e:
        logger.warning(f"Could not optimize CPU settings: {e}")