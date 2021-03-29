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

def constrastMatrix(matrix):
    support_matrix = np.fromfunction(lambda i, j: (i-j)**2, shape = matrix.shape,)
    
    return np.multiply(matrix, support_matrix).sum()

def getMeans(matrix):
    mean_x = matrix.sum(axis=1) @ np.array(range(len(matrix)))
    mean_y = matrix.sum(axis=0) @ np.array(range(len(matrix)))
    
    return mean_x, mean_y

def getStd(matrix, mean_x, mean_y):
    std_x = matrix.sum(axis=1) @ ((np.array(range(len(matrix))) - mean_x)**2)
    std_y = matrix.sum(axis=0) @ ((np.array(range(len(matrix))) - mean_y)**2)
    
    return std_x**(0.5), std_y**(0.5)

def correlationMatrix_(matrix, mean_x,  mean_y, std_x, std_y):
    support_i = np.fromfunction(lambda i, j: i + 1, shape = matrix.shape,)
    support_j = np.fromfunction(lambda i, j: j + 1, shape = matrix.shape,)
    
    support_i = support_i - mean_x
    support_j = support_j - mean_y
    
    return np.multiply(np.multiply(support_i, support_j), matrix).sum() / (std_x * std_y)

def getCorrelation(matrix):
    mean_x, mean_y = getMeans(matrix)
    std_x, std_y = getStd(matrix, mean_x, mean_y)
    correlation_value = correlationMatrix_(matrix, mean_x,  mean_y, std_x, std_y)
    
    return correlation_value

def variance(i,j):
    return ((i+1))

def varianceMatrix(matrix):
    mu = np.sum(matrix)/(len(matrix)**2)
    support_matrix = np.fromfunction(variance, shape = matrix.shape, )
    support_matrix = support_matrix - mu
    support_matrix = support_matrix**2
    
    return np.multiply(matrix, support_matrix).sum()

def inverseDifferenceMoment(i, j):
    return 1/(1 + ((i-j)**2))

def inverseDifferenceMomentMatrix(matrix):
    support_matrix = np.fromfunction(inverseDifferenceMoment, shape = matrix.shape, )
    
    return np.multiply(matrix, support_matrix).sum()

def entropyMatrix(matrix):
    e = 1e-30
    
    return np.multiply(matrix, np.log(matrix + e)).sum()

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

def Px_y(matrix):
    array = []
    
    for i in range(len(matrix)):
        if i == 0:
            array.append(np.trace(matrix, offset=i,))
        else:
            array.append(np.trace(matrix, offset=i,) + np.trace(matrix, offset=-i,))
    
    return np.array(array)

def MUx_y(matrix, px_y):
    n = np.array(range(len(matrix)))    
    return n @ px_y

def differenceVariance(matrix):
    p_xy = Px_y(matrix)
    mu_xy = MUx_y(matrix, p_xy)
    
    n = list(range(len(matrix)))
    n = [(x-mu_xy)**2 for x in n]
    n = np.array(n)
    
    return -(n @ p_xy)

def differenceEntropy(matrix):
    e = 1e-30
    p_xy = Px_y(matrix)
    
    return p_xy @ (p_xy + e)