
import numpy as np
import math

# Grayscale
def grayscale(rgb_image):
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b 
    return gray_image

def conv(img, filter):
    irows, icols = img.shape
    frows, fcols = filter.shape
    rrows, rcols = irows - frows + 1, icols - fcols + 1
    result = np.zeros( (rrows, rcols) )

    for i in range(rrows):
        for j in range(rcols):
                    result[i][j] = np.sum( img[i:i+frows, j:j+fcols] * filter ) 
    
    return result

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
    
    return blurred

def get_gradients(img):
    Gx = np.array([ 
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]
    ])

    Gy = np.array([
                    [1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]
    ])

    grad_x = conv(img, Gx)
    grad_y = conv(img, Gy)
    rows, cols = grad_x.shape
    grad = np.zeros((rows,cols)) 
    for i in range(rows):
        for j in range(cols):
            grad[i][j] = abs(grad_x[i][j]) + abs(grad_y[i][j])

    return grad_x, grad_y, grad

def get_gradient_angles(y,x):
    rows, cols = y.shape
    result = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if x[i][j] == 0:
                result[i][j] = 0
            else:    
                result[i][j] =  np.degrees( np.arctan(y[i][j] / x[i][j]) )

    return result

def check_sector(x):
    if (x <= 22.5 and x>= -22.5):
        return 0
    elif x > 22.5 and x <= 67.5:
        return 1
    elif x < -22.5 and x >= -67.5:
        return 3
    else:
        return 2 

def nms(img, angles):
    
    rows, cols = img.shape
    #result = img[1:rows-1][1:cols-1]

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            sector = check_sector(angles[i][j])
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

def thresholding(img):
    all = img.flatten().tolist()
    arr = list(set(all))
    #arr = []
  

    print("all ", min(all) , max(all))
    print( "Total ",len(arr) )
    print( "Non zero ",np.count_nonzero(arr) )
    p_25th = np.percentile(arr, 25)
    p_50th = np.percentile(arr, 50)
    p_75th = np.percentile(arr, 75)

    rows, cols = img.shape
    result_25th = np.zeros( (rows, cols) )
    result_50th = np.zeros( (rows, cols) )
    result_75th = np.zeros( (rows, cols) )

    print(p_25th, p_50th, p_75th)

    for i in range(rows):
        for j in range(cols):
            if img[i][j] >= p_75th:
                result_75th[i][j] = 255
            
            if img[i][j] >= p_50th:
                result_50th[i][j] = 255
            
            if img[i][j] >= p_25th:
                result_25th[i][j] = 255               



    return result_25th, result_50th, result_75th


