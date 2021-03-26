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
            ## ES NECESARIO SUMARLE 1 A LOS ÍNDICES??? NO LO SÉ
            matrixSum += ((i+1) - mean_x) * ((j+1) - mean_y) * matrix[i][j] 
    
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

def variance(matrix):
    mu = np.sum(matrix)/(len(matrix)**2)
    variance = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ## ES NECESARIO SUMARLE 1 A LOS ÍNDICES??? NO LO SÉ
            variance += (((i+1) - mu)**2) * (matrix[i][j])
    
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

def sumAverageMatrix(matrix):
    sumvector = []
    new_matrix = np.rot90(matrix.copy(), k=3, axes=(0, 1))
    indices = list(range(-len(matrix)+1, len(matrix)))
    indices.reverse()
    
    n = np.array(range(2, len(matrix)*2 + 1))
    
    for i in indices:
        sumvector.append(np.trace(new_matrix, offset=i,))
    
    sumvector = np.array(sumvector)
    
    return n @ sumvector

def sumVarianceMatrix(matrix, sumaverage):
    sumvector = []
    new_matrix = np.rot90(matrix.copy(), k=3, axes=(0, 1))
    indices = list(range(-len(matrix)+1, len(matrix)))
    indices.reverse()
    
    n = list(range(2, len(matrix)*2 + 1))
    n = [(x-sumaverage)**2 for x in n]
    n = np.array(n)
    
    for i in indices:
        sumvector.append(np.trace(new_matrix, offset=i,))
    
    sumvector = np.array(sumvector)
    
    return n @ sumvector

def sumEntropyMatrix(matrix):
    e = 1e-30
    sumentropy = 0
    
    new_matrix = np.rot90(matrix.copy(), k=3, axes=(0, 1))
    indices = list(range(-len(matrix)+1, len(matrix)))
    indices.reverse()
    
    for i in indices:
        intermediate_sum = np.trace(new_matrix, offset=i,)
        sumentropy += intermediate_sum * np.log(intermediate_sum + e)
    
    return -sumentropy