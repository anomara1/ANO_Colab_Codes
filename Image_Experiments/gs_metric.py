import cv2
import numpy as np

def compute_gradients(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients using Sobel operator
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return grad_magnitude

def compute_gradient_similarity(image1, image2):
    # Compute gradients for both images
    grad_magnitude1 = compute_gradients(image1)
    grad_magnitude2 = compute_gradients(image2)
    
    # Normalize the gradients to [0, 1] range
    grad_magnitude1 = grad_magnitude1 / np.max(grad_magnitude1)
    grad_magnitude2 = grad_magnitude2 / np.max(grad_magnitude2)
    
    # Compute the similarity using cosine similarity between gradients
    similarity = np.sum(grad_magnitude1 * grad_magnitude2) / (np.linalg.norm(grad_magnitude1) * np.linalg.norm(grad_magnitude2))
    
    return similarity

# Example usage
image1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

similarity = compute_gradient_similarity(image1, image2)
print(f'Gradient Similarity Score: {similarity}')
