import cv2
import numpy as np
import sys
from utils import *

path = sys.argv[1] # Get the location of the image
img = cv2.imread(path, 0) # Read the image as a grayscale image
img = np.array( img ) # Convert the image to a numpy array

print("Input image shape: ", img.shape) # Input image shape

cv2.imwrite("output (0) Gray.bmp", img) # Save grayscale version of the input image

# 1. Apply Gaussian blur
img = Gaussian_blur( img )
print("Blurred image size: ",img.shape)
cv2.imwrite("output (1) Blurred.bmp", img) # Save blurred image

# 2. Compute gradients
grad_x, grad_y, grad = compute_gradients(img)
print("Gradients shape: ", grad_x.shape, grad_y.shape, grad.shape)
cv2.imwrite("output (2x).bmp", grad_x) # Gx output
cv2.imwrite("output (2y).bmp", grad_y) # Gy output
cv2.imwrite("output (2z).bmp", grad) # Gradient magnitude output


# Compute Gradient angles
grad_angles = compute_gradient_angles(grad_y, grad_x)

# 3. Non Maxima Suppression 
img = nms(grad, grad_angles)
print("NMS output shape: ", img.shape)
cv2.imwrite("output (nms).bmp", img)

# 4. Thresholding
img = np.matrix.clip(img, 0, 255) # Clip values exceeding 255
r25, r50, r75 = thresholding(img) 
print("Final output image shape: ", r25.shape)
cv2.imwrite("output (25%).bmp", r25)
cv2.imwrite("output (50%).bmp", r50)
cv2.imwrite("output (75%).bmp", r75) # Save the final output images
