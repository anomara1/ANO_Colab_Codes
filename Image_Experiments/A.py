!pip install piq
!pip install sewar
import time
import numpy as np
import matplotlib.pyplot as plt
from piq import brisque, fid, gs, haarpsi, iw_ssim,msid,kid,pieapp,vsi
import sewar
from skimage import data, img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.spatial.distance import euclidean, cityblock, cosine, chebyshev, minkowski, braycurtis
from scipy.stats import entropy
from skimage.color import rgb2gray, rgba2rgb
from scipy.linalg import norm
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.util import random_noise
from tensorflow.keras.datasets import mnist
import cv2
import compute_image_quality_metrics

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
    #["MDSI"] = compute_mdsi(original, modified)
    metrics["Gradient Similarity"] = compute_gs(original, modified)
    #metrics["NIQE"] = niqe(modified_uint8)
    #Qmetrics["PIQE"] = sewar.piqe(modified_uint8)
    #metrics["BRISQUE"] = brisque(modified_uint8)
    #metrics["FID"] = fid(original_uint8, modified_uint8)
    #metrics["GS"] = gs(original_uint8, modified_uint8)
    #metrics["HAARPSI"] = haarpsi(original_uint8, modified_uint8)
    #metrics["IW-SSIM"] = iw_ssim(original_uint8, modified_uint8)
    #metrics["MSID"] = msid(original_uint8, modified_uint8)
    #metrics["KID"] = kid(original_uint8, modified_uint8)
    #metrics["PIEAPP"] = pieapp(original_uint8, modified_uint8)
    #metrics["SRSIM"] = srsim(original_uint8, modified_uint8)

    # General Metrics
    metrics["PSNR"] = psnr(original, modified, data_range=1)
    metrics["MSE"] = np.mean((original - modified) ** 2)
    metrics["Euclidean"] = euclidean(original.ravel(), modified.ravel())
    metrics["Manhattan"] = cityblock(original.ravel(), modified.ravel())
    metrics["Cosine"] = cosine(original.ravel(), modified.ravel())
    metrics["Chebyshev"] = chebyshev(original.ravel(), modified.ravel())
    metrics["Minkowski"] = minkowski(original.ravel(), modified.ravel(), p=3)
    metrics["Bray-Curtis"] = braycurtis(original.ravel(), modified.ravel())
    metrics["KL Divergence"] = entropy(original.ravel() + 1e-10, modified.ravel() + 1e-10)
    metrics["Shannon Entropy"] = shannon_entropy(modified)
    metrics["Total Variation"] = np.sum(np.abs(original - modified))
    metrics["Gradient Magnitude"] = norm(sobel(original) - sobel(modified))

    elapsed_time = time.time() - start_time
    metrics["Time (s)"] = elapsed_time

    return metrics

# Define Image Categories
categories = {
    "Natural": [preprocess_image(data.camera()), preprocess_image(data.astronaut()), preprocess_image(data.coffee())],
    "Objects": [preprocess_image(data.checkerboard()), preprocess_image(data.rocket()), preprocess_image(data.moon())]
}

# Define Noise Levels
noise_levels = np.linspace(0.001, 0.1, 10)

# Initialize Results
metric_names = list(compute_metrics(np.zeros((10, 10), dtype=np.float32), np.zeros((10, 10), dtype=np.float32)).keys())
results = {category: {key: np.zeros(len(noise_levels)) for key in metric_names} for category in categories}

# Compute Metrics Across Noise Levels
for category, images in categories.items():
    for image in images:
        for i, var in enumerate(noise_levels):
            noisy_image = apply_gaussian_noise(image, var)
            metrics = compute_metrics(image, noisy_image)
            for key in results[category]:
                results[category][key][i] += metrics[key]

# Normalize Results
for category in results:
    for key in results[category]:
        results[category][key] /= len(categories[category])

# Plot the Results
fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(15, 20))
axes = axes.ravel()

for i, key in enumerate(results["Natural"]):
    for category in results:
        axes[i].plot(noise_levels, results[category][key], marker='o', label=category)
    axes[i].set_title(key)
    axes[i].set_xlabel("Gaussian Noise Variance")
    axes[i].set_ylabel("Average Metric Value")
    axes[i].legend()
    axes[i].grid()

# Remove Empty Subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("image_noise_quality_metrics.png")
plt.show()
