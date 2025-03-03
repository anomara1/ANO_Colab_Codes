# compute_image_quality_metrics.py

import time
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import sobel
from skimage.util import random_noise
import sewar

# Image Preprocessing
def preprocess_image(image):
    image = img_as_float(image)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        image = rgb2gray(image)
    return image

# Apply Gaussian Noise
def apply_gaussian_noise(image, var):
    return random_noise(image, mode='gaussian', var=var)

# Compute Gradient-Based Features
def compute_phase_congruency(image):
    grad_x = sobel(image)
    grad_y = sobel(image)
    return np.sqrt(grad_x**2 + grad_y**2)

# Compute Feature Similarity Index (FSIM)
def compute_fsim(original, modified):
    pc_original = compute_phase_congruency(original)
    pc_modified = compute_phase_congruency(modified)
    grad_original = sobel(original)
    grad_modified = sobel(modified)
    similarity_map = ((2 * pc_original * pc_modified + 1e-10) /
                      (pc_original**2 + pc_modified**2 + 1e-10)) * \
                     ((2 * grad_original * grad_modified + 1e-10) /
                      (grad_original**2 + grad_modified**2 + 1e-10))
    return float(np.mean(similarity_map))

# Compute Gradient Magnitude Similarity Deviation (GMSD)
def compute_gmsd(original, modified):
    original_grad = sobel(original)
    modified_grad = sobel(modified)
    gms = (2 * original_grad * modified_grad + 1e-10) / (original_grad**2 + modified_grad**2 + 1e-10)
    return float(np.std(gms))

# Compute Additional Metrics
def compute_vsi(original, modified):
    return float(sewar.vsi(img_as_ubyte(original), img_as_ubyte(modified)))

def compute_vif(original, modified):
    return float(sewar.vifp(img_as_ubyte(original), img_as_ubyte(modified)))

def compute_mdsi(original, modified):
    return float(sewar.mdsi(img_as_ubyte(original), img_as_ubyte(modified)))

def compute_gs(original, modified):
    original_grad = sobel(original)
    modified_grad = sobel(modified)
    gs_map = (2 * original_grad * modified_grad + 1e-10) / (original_grad**2 + modified_grad**2 + 1e-10)
    return float(np.mean(gs_map))

# Compute Image Quality Metrics
def compute_metrics(original, modified):
    start_time = time.time()
    metrics = {}

    original_uint8 = img_as_ubyte(original)
    modified_uint8 = img_as_ubyte(modified)

    # Image-Specific Metrics
    metrics["SSIM"] = ssim(original, modified, data_range=1)
    metrics["FSIM"] = compute_fsim(original, modified)
    metrics["GMSD"] = compute_gmsd(original, modified)
    metrics["VSI"] = compute_vsi(original, modified)
    metrics["VIF"] = compute_vif(original, modified)
    metrics["Gradient Similarity"] = compute_gs(original, modified)

    # General Metrics
    metrics["PSNR"] = psnr(original, modified, data_range=1)
    metrics["MSE"] = np.mean((original - modified) ** 2)

    elapsed_time = time.time() - start_time
    metrics["Time (s)"] = elapsed_time

    return metrics
