import os
import torch
import tensorflow as tf
from tensorflow import keras
from diffusers import UNet2DModel, DDIMScheduler
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Global model instances
cnn_model = None
diffusion_model = None
device = None

# --- PROVIDED LOSS AND METRIC FUNCTIONS (Must be included) --- #
# Dice Loss
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# Dice Coefficient
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Combined Weighted Binary Cross-Entropy and Dice Loss
def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5, class_weights=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if class_weights is not None:
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')(y_true, y_pred)
        weight_mask = tf.cast(tf.where(y_true == 1, class_weights[1], class_weights[0]), tf.float32)
        weight_mask = tf.squeeze(weight_mask, axis=-1)
        weighted_bce = tf.reduce_mean(bce * weight_mask)
    else:
        weighted_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss_val = 1 - dice
    return alpha * weighted_bce + beta * dice_loss_val

# --- U-Net Model Definition ---
def build_unet_model(input_shape):
    inputs = keras.Input(input_shape)
    # Contracting Path (Encoder)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = keras.layers.Dropout(0.1)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = keras.layers.Dropout(0.2)(c3)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = keras.layers.Dropout(0.2)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    # Bridge
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = keras.layers.Dropout(0.3)(c5)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    # Expanding Path (Decoder)
    u6 = keras.layers.UpSampling2D((2, 2))(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = keras.layers.Dropout(0.2)(c6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    u7 = keras.layers.UpSampling2D((2, 2))(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = keras.layers.Dropout(0.2)(c7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    u8 = keras.layers.UpSampling2D((2, 2))(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = keras.layers.Dropout(0.1)(c8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    u9 = keras.layers.UpSampling2D((2, 2))(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = keras.layers.Dropout(0.1)(c9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- The function to load the weights-only model ---
def load_cnn_model_weights_only(model_path):
    """Load the CNN model from a weights-only file using the provided architecture and losses."""
    try:
        # Force CPU for TF
        tf.config.set_visible_devices([], 'GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Build the architecture first
        input_shape = (256, 256, 1)
        model = build_unet_model(input_shape)

        # Load weights
        model.load_weights(model_path)

        # Compile for inference (required by Keras for predict)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coef])

        logger.info("✅ CNN model loaded successfully from weights-only file")
        return model
    except Exception as e:
        logger.error(f"❌ Could not load CNN model: {e}")
        return None


def load_models(cnn_model_path, diffusion_model_path):
    """
    Load both CNN (Keras) and Diffusion (PyTorch) models for CPU only
    Returns True only if BOTH models load successfully
    """
    global cnn_model, diffusion_model, device

    # Force CPU usage
    device = "cpu"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for TensorFlow

    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')

    logger.info(f"Using device: {device} (CPU only)")

    cnn_success = False
    diffusion_success = False

    # Try to load CNN model (weights-only)
    try:
        logger.info(f"Loading CNN model from {cnn_model_path}")
        cnn_model = load_cnn_model_weights_only(cnn_model_path)
        if cnn_model is not None:
            logger.info("✅ CNN model loaded successfully")
            cnn_success = True
        else:
            raise RuntimeError("CNN model returned None")
    except Exception as e:
        logger.error(f"❌ Failed to load CNN model: {e}")
        cnn_model = None

    # Try to load Diffusion model (PyTorch) - independent of CNN loading
    try:
        logger.info(f"Loading diffusion model from {diffusion_model_path}")

        # Define the model architecture to match the trained model exactly
        diffusion_model = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),  # Corrected: last block is 256, not 512
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),  # Only block 1 has attention
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),  # Only block 1 has attention
            attention_head_dim=8,
        )

        # Load the weights onto CPU
        checkpoint = torch.load(diffusion_model_path, map_location=torch.device('cpu'))
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        diffusion_model.to(device)
        diffusion_model.eval()
        logger.info("✅ Diffusion model loaded successfully")
        diffusion_success = True

    except Exception as e:
        logger.error(f"❌ Failed to load diffusion model: {e}")
        diffusion_model = None

    # Return True only if BOTH models loaded successfully
    success = cnn_success and diffusion_success

    if success:
        logger.info("🎉 Both models loaded successfully!")
    else:
        logger.warning(f"⚠️ Partial loading: CNN={cnn_success}, Diffusion={diffusion_success}")

    return success


def get_models():
    """Return loaded models"""
    return cnn_model, diffusion_model, device


def models_loaded():
    """Check if models are loaded"""
    return cnn_model is not None and diffusion_model is not None
