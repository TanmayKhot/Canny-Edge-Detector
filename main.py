import cv2
import numpy as np
import sys
from utils import *

path = sys.argv[1]
img = cv2.imread(path)
img = np.array( img )

h,w,c = img.shape

# If RGB value then convert to grayscale
if c>1:
    img = grayscale(img)

cv2.imwrite("output.jpg", img)

