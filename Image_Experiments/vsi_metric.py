import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def compute_luminance(image):
    # If the image is grayscale, return the image itself as luminance
    if image.ndim == 2:
        return image
    # Otherwise, calculate the mean along the color channel axis
    else:
        return np.mean(image, axis=2)

def compute_contrast(image, luminance):
    return np.sqrt(np.mean((image - luminance[:, :, None]) ** 2, axis=2))

def compute_structure(image, luminance, contrast):
    return (image - luminance[:, :, None]) / contrast[:, :, None]

def compute_vsi(image1, image2):
    # Convert images to grayscale if they are in color
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
    if len(image1.shape) == 2:  # Grayscale image
        image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
        image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)

    # Apply Gaussian filtering to smooth the images
    image1 = gaussian_filter(image1, sigma=1.5)
    image2 = gaussian_filter(image2, sigma=1.5)
    
    # Compute luminance, contrast, and structure for both images
    luminance1 = compute_luminance(image1)
    luminance2 = compute_luminance(image2)
    
    contrast1 = compute_contrast(image1, luminance1)
    contrast2 = compute_contrast(image2, luminance2)
    
    structure1 = compute_structure(image1, luminance1, contrast1)
    structure2 = compute_structure(image2, luminance2, contrast2)
    
    # Compute the similarity between the luminance, contrast, and structure
    luminance_similarity = np.exp(-np.mean((luminance1 - luminance2) ** 2))
    contrast_similarity = np.exp(-np.mean((contrast1 - contrast2) ** 2))
    structure_similarity = np.exp(-np.mean((structure1 - structure2) ** 2))
    
    # Combine the similarities to get the final VSI score
    vsi_score = (luminance_similarity + contrast_similarity + structure_similarity) / 3
    return vsi_score

# Example usage
#image1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
#image2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

#vsi_score = compute_vsi(image1, image2)
#print(f'VSI Score: {vsi_score}')
