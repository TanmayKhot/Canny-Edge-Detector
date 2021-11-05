
import numpy as np
import math

# Padding
# - Using zero padding
# - Default value is 1 because of convolution operations with 3x3 filter
# - pad_size will be 3 only for padding the output of Gaussian Blurring -
#   since it is using a 7x7 filter
def pad(img, pad_size=1):
    h,w = img.shape
    padded_img = np.zeros((h + 2*pad_size, w + 2*pad_size))
    
    for i in range(pad_size,h+pad_size):
        for j in range(pad_size,w+pad_size):
                padded_img[i][j] = img[i-pad_size][j-pad_size]
    return padded_img

# Convolution 
def conv(img, filter):
    irows, icols = img.shape # image
    frows, fcols = filter.shape # filter
    rrows, rcols = irows - frows + 1, icols - fcols + 1 # result
    result = np.zeros( (rrows, rcols) )

    for i in range(rrows):
        for j in range(rcols):
                    result[i][j] = np.sum( img[i:i+frows, j:j+fcols] * filter ) 
    
    return result

# Gaussian Blur
# Will reduce the output image by 3 pixels from every direction because
# of 7x7 filter

def Gaussian_blur(img):
    Gaussian = np.array([ 
                    [ 1, 1, 2, 2, 2, 1, 1],
                    [ 1, 2, 2, 4, 2, 2, 1],
                    [ 2, 2, 4, 8, 4, 2, 2],
                    [ 2, 4, 8, 16, 8, 4, 2],
                    [ 1, 2, 2, 4, 2, 2, 1],
                    [ 2, 2, 4, 8, 4, 2, 2],
                    [ 1, 1, 2, 2, 2, 1, 1],
    ])

    blurred = conv(img, Gaussian)
    rows,cols = blurred.shape
    for i in range(rows):
        for j in range(cols):
            blurred[i][j] /= 140

    return pad(blurred,3) # Padding the output image to maintain the original size
    

# Compute gradients
def compute_gradients(img): 
    Gx = np.array([ 
                    [-1, 0, 1], # Prewitt's operator for Gradients Gx
                    [-1, 0, 1],
                    [-1, 0, 1]
    ]) 

    Gy = np.array([
                    [1, 1, 1], # Prewitt's operator for Gradients Gy
                    [0, 0, 0],
                    [-1, -1, -1]
    ])

    grad_x = pad(conv(img, Gx))
    grad_y = pad(conv(img, Gy))
    
    x, y = grad_x.shape
    pad_x, pad_y = np.zeros((x,y)), np.zeros((x,y))
    
    for i in range(4,x-4):  # Ignoring the border pixels from padding from previous operations
        for j in range(4,y-4):
                pad_x[i][j] = grad_x[i][j]
                pad_y[i][j] = grad_y[i][j]

    rows, cols = grad_x.shape
    grad = np.zeros((rows,cols)) 
    for i in range(rows):
        for j in range(cols):
            grad[i][j] = abs(pad_x[i][j]) + abs(pad_y[i][j])

    return pad_x, pad_y, grad

# Computer Gradient angles
def compute_gradient_angles(y,x):
    rows, cols = y.shape
    result = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if x[i][j] != 0:   # to avoid division by zero
                result[i][j] =  np.degrees( np.arctan(y[i][j] / x[i][j]) )

    return result

# Finding sector for NMS
def check_sector(x):
    if (x <= 22.5 and x>= -22.5):
        return 0
    elif x > 22.5 and x <= 67.5:
        return 1
    elif x < -22.5 and x >= -67.5:
        return 3
    else:
        return 2 

# Non Maximum Suppression
def nms(img, angles): # minus 2
    
    rows, cols = img.shape
    #result = img[1:rows-1][1:cols-1]

    for i in range(5,rows-5): #  Ignoring first and last 5 pixels from previous operations
        for j in range(5,cols-5):  
                sector = check_sector(angles[i][j])
                # If the pixel is not a local maxima then suppress
                if sector == 0:
                    if img[i][j] < img[i][j-1] or img[i][j] < img[i][j+1]:
                        img[i][j] = 0 
                if sector == 1:
                    if img[i][j] < img[-1][j+1] or img[i][j] < img[i+1][j-1]:
                        img[i][j] = 0
                if sector == 3:
                    if img[i][j] < img[i-1][j-1] or img[i][j] < img[i+1][j+1]:
                        img[i][j] = 0
                if sector == 2:
                    if img[i][j] < img[i-1][j] or img[i][j] < img[i+1][j]:
                        img[i][j] = 0
    
    return img

# Thresholding
def thresholding(img):
    arr = img.flatten().tolist()
    arr = list(set(arr))
    arr.remove(0) # Removing 0 before computing thresholds
    
    # Compute percentiles
    p_25th = np.percentile(arr, 25)
    p_50th = np.percentile(arr, 50)
    p_75th = np.percentile(arr, 75)

    rows, cols = img.shape
    result_25th = np.zeros( (rows, cols) )
    result_50th = np.zeros( (rows, cols) )
    result_75th = np.zeros( (rows, cols) )

    for i in range(rows):
        for j in range(cols):
            # Keep the pixels above a certain threshold
            if img[i][j] > p_75th:
                result_75th[i][j] = 255
            
            if img[i][j] > p_50th:
                result_50th[i][j] = 255
            
            if img[i][j] > p_25th:
                result_25th[i][j] = 255               



    return  result_25th, result_50th, result_75th


