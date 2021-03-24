import numpy as np
from skimage.feature import greycomatrix

def horizontalGTSDM(window, levels):
    result = greycomatrix(window, [1], [0], levels=levels, normed=True, symmetric=True)
    
    return result[:, :, 0, 0]

def verticalGTSDM(window, levels):
    result = greycomatrix(window, [1], [np.pi/2], levels=levels, normed=True, symmetric=True)
    
    return result[:, :, 0, 0]

def l_diagonalGTSDM(window, levels):
    result = greycomatrix(window, [1], [np.pi/4], levels=levels, normed=True, symmetric=True)
    
    return result[:, :, 0, 0]

def r_diagonalGTSDM(window, levels):
    result = greycomatrix(window, [1], [3*np.pi/4], levels=levels, normed=True, symmetric=True)
    
    return result[:, :, 0, 0]

def ASM(matrix):
    return np.sum(matrix**2)

def constrast(matrix):
    matrixSum = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrixSum += ((i-j)**2)*matrix[i][j]
    
    return matrixSum

def correlation_(matrix, mean_x,  mean_y, std_x, std_y):
    matrixSum = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrixSum += (i - mean_x) * (j - mean_y) * matrix[i][j]
    
    return matrixSum/(std_x * std_y)

def getMeans(matrix):
    mean_x = matrix.sum(axis=1) @ np.array(range(len(matrix)))
    mean_y = matrix.sum(axis=0) @ np.array(range(len(matrix)))
    
    return mean_x, mean_y

def getStd(matrix, mean_x, mean_y):
    std_x = matrix.sum(axis=1) @ ((np.array(range(len(matrix))) - mean_x)**2)
    std_y = matrix.sum(axis=0) @ ((np.array(range(len(matrix))) - mean_y)**2)
    
    return std_x**(0.5), std_y**(0.5)

def getCorrelation(matrix):
    mean_x, mean_y = getMeans(matrix)
    std_x, std_y = getStd(matrix, mean_x, mean_y)
    correlation_value = correlation_(matrix, mean_x,  mean_y, std_x, std_y)
    
    return correlation_value

def variance(matrix, level):
    mu = np.sum(matrix)/(level**2)
    variance = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            variance += ((i - mu)**2) * (matrix[i][j])
    
    return variance

def inverseDifferenceMoment(matrix):
    idm = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            idm += (matrix[i][j]) / (1 + ((i-j)**2))
    
    return idm

def entropy(matrix):
    e = 1e-30
    entropy = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            entropy += (matrix[i][j]) * np.log(matrix[i][j] + e)
    
    return -entropy