import cv2
import numpy as np
import os
from skimage.feature import greycomatrix
import pandas as pd
from multiprocessing import Pool
import multiprocessing

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

def getDictMatrices(matrix, window_size):
    row_first = 0
    row_last = window_size
    center = int((window_size-1)/2)
    dict_of_matrices = []
    keys = []

    for i in range(matrix.shape[0] + 1):
        
        if row_last == matrix.shape[1] + 1:
            break
        
        column_first = 0
        column_last = window_size
        for j in range(matrix.shape[1] + 1):
            
            if column_last == matrix.shape[1] + 1:
                break
            
            window = matrix[row_first:row_last, column_first:column_last]
            key_name = 'cell_%s_%s'%(row_first + center, column_first + center)
            dict_of_matrices.append(window)
            keys.append(key_name)
            column_first+=1
            column_last+=1

        row_first+=1
        row_last+=1
    
    return dict_of_matrices, keys

def processMatrix_horizontal(matrix):
    image_horizontal = horizontalGTSDM(matrix, 256)

    return [ASM(image_horizontal), constrast(image_horizontal), getCorrelation(image_horizontal)]

def processMatrix_vertical(matrix):
    image_vertical = verticalGTSDM(matrix, 256)

    return [ASM(image_vertical), constrast(image_vertical), getCorrelation(image_vertical)]

def processMatrix_l_diagonal(matrix):
    image_l_diagonal = l_diagonalGTSDM(matrix, 256)

    return [ASM(image_l_diagonal), constrast(image_l_diagonal), getCorrelation(image_l_diagonal)]

def processMatrix_r_diagonal(matrix):
    image_r_diagonal = r_diagonalGTSDM(matrix, 256)

    return [ASM(image_r_diagonal), constrast(image_r_diagonal), getCorrelation(image_r_diagonal)]

images_path = '/home/nicolas/Documents/tesis/pca_imagenes/'
csv_path = '/home/nicolas/Documents/tesis/haralick_features/'
images = os.listdir()
num_cores = multiprocessing.cpu_count()

spatial_dependence_matrix = {'horizontal': processMatrix_horizontal,
                            'vertical': processMatrix_vertical,
                            'l_diagonal': processMatrix_l_diagonal,
                            'r_diagonal': processMatrix_r_diagonal}

chanels = { 'blue': blue,
            'green': green,
            'red': red}

for image in images:
    try:
        os.chdir(images_path)
        pic = cv2.imread(image, cv2.IMREAD_COLOR)
        blue, green, red = cv2.split(pic)
        for chanel in chanels:
            matrix = chanels[chanel]
            test = matrix.copy()[0:100, 0:100]
            list_of_matrices, keys = getDictMatrices(test, 5)
            for method in spatial_dependence_matrix:
                pool = Pool(num_cores)
                result = pool.map(func=spatial_dependence_matrix[method], iterable = list_of_matrices,)
                data_in_dict = {keys[i]: result[i] for i in range(len(keys))}
                data_in_pandas = pd.DataFrame.from_dict(data_in_dict, orient='index', 
                                                        columns=['ASM', 'contrast', 'correlation'])
                pool.terminate()
                
                os.chdir(csv_path)
                csv_name = '%s_%s_%s.csv'%(image, chanel, method)
                data_in_pandas.reset_index().to_csv(csv_name, index=False)
    except:
        pass