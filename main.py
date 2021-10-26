import cv2
import numpy as np
import sys
from utils import *

# python main.py *path to the image.bmp*

path = sys.argv[1]
img = cv2.imread(path)
img = np.array( img )

h,w,c = img.shape

# 0. If RGB value then convert to grayscale
if c>1:
    img = grayscale(img)

print("grayscaled")
cv2.imwrite("output (0).jpg", img)

# 1. Apply Gaussian blur
img = Gaussian_blur(img)
print("Blurred")
cv2.imwrite("output (1).jpg", img)

# 2. Compute gradients
grad_x, grad_y, grad = get_gradients(img)
print("Computed gradients")
cv2.imwrite("output (2x).jpg", grad_x)
cv2.imwrite("output (2y).jpg", grad_y)
cv2.imwrite("output (2z).jpg", grad)

# Compute Gradient angles
grad_angles = get_gradient_angles(grad_y, grad_x)
print("Computed angles")

# 3. Non Maxima Suppresion 
img = nms(grad, grad_angles)
print("completed NMS")

img = np.matrix.clip(img, 0, 255)
print("MinMax ", img.min(), img.max())

r25, r50, r75 = thresholding(img)
print("Completed thresholding")
cv2.imwrite("output (25%).jpg", r25)
cv2.imwrite("output (50%).jpg", r50)
cv2.imwrite("output (75%).jpg", r75)







