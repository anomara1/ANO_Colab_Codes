import numpy as np
import cv2
import pywt
import pywt.data

def haar_wavelet_transform(image):
    """
    Perform Haar Wavelet Transform on the image.
    Returns the approximation (low-frequency) and detail (high-frequency) coefficients.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform 2D Haar wavelet transform
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2  # Low-low, Low-high, High-low, High-high coefficients
    return LL, LH, HL, HH

def compute_haarpsi(image1, image2):
    """
    Compute HAARPSI score between two images.
    The lower the score, the higher the image quality.
    """
    # Perform Haar wavelet transform on both images
    LL1, LH1, HL1, HH1 = haar_wavelet_transform(image1)
    LL2, LH2, HL2, HH2 = haar_wavelet_transform(image2)
    
    # Calculate the mean squared error (MSE) for wavelet coefficients
    mse_LL = np.mean((LL1 - LL2) ** 2)
    mse_LH = np.mean((LH1 - LH2) ** 2)
    mse_HL = np.mean((HL1 - HL2) ** 2)
    mse_HH = np.mean((HH1 - HH2) ** 2)
    
    # Combine the MSEs to compute the overall score (a lower score indicates better similarity)
    score = mse_LL + mse_LH + mse_HL + mse_HH
    return score

# Example usage
image1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

# Compute HAARPSI score
#haarpsi_score = compute_haarpsi(image1, image2)
#print(f'HAARPSI Score: {haarpsi_score}')
