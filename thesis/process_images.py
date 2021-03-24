import cv2
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import time
import haralick_features as hf

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
    image_horizontal = hf.horizontalGTSDM(matrix, 256)

    return [hf.ASM(image_horizontal), hf.constrast(image_horizontal), hf.getCorrelation(image_horizontal)]

def processMatrix_vertical(matrix):
    image_vertical = hf.verticalGTSDM(matrix, 256)

    return [hf.ASM(image_vertical), hf.constrast(image_vertical), hf.getCorrelation(image_vertical)]

def processMatrix_l_diagonal(matrix):
    image_l_diagonal = hf.l_diagonalGTSDM(matrix, 256)

    return [hf.ASM(image_l_diagonal), hf.constrast(image_l_diagonal), hf.getCorrelation(image_l_diagonal)]

def processMatrix_r_diagonal(matrix):
    image_r_diagonal = hf.r_diagonalGTSDM(matrix, 256)

    return [hf.ASM(image_r_diagonal), hf.constrast(image_r_diagonal), hf.getCorrelation(image_r_diagonal)]

def tac(_start_time):
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {} hour - {} min - {} sec'.format(t_hour,t_min,t_sec))

images_path = '/home/nicolas/Documents/tesis/pca_imagenes/'
csv_path = '/home/nicolas/Documents/tesis/haralick_features/'
os.chdir(images_path)
images = os.listdir()
num_cores = multiprocessing.cpu_count()

spatial_dependence_matrix = {'horizontal': processMatrix_horizontal,
                            'vertical': processMatrix_vertical,
                            'l_diagonal': processMatrix_l_diagonal,
                            'r_diagonal': processMatrix_r_diagonal}

for image in images:
    try:
        os.chdir(images_path)
        pic = cv2.imread(image, cv2.IMREAD_COLOR)
        blue, green, red = cv2.split(pic)
        chanels = { 'blue': blue,
                    'green': green,
                    'red': red}
        for chanel in chanels:
            matrix = chanels[chanel]
            list_of_matrices, keys = getDictMatrices(matrix, 7)
            for method in spatial_dependence_matrix:
                _start_time = time.time()
                
                pool = Pool(num_cores)
                result = pool.map(func=spatial_dependence_matrix[method], iterable = list_of_matrices,)
                data_in_dict = {keys[i]: result[i] for i in range(len(keys))}
                data_in_pandas = pd.DataFrame.from_dict(data_in_dict, orient='index', 
                                                        columns=['ASM', 'contrast', 'correlation'])
                pool.terminate()
                
                os.chdir(csv_path)
                csv_name = '%s_%s_%s.csv'%(image, chanel, method)
                data_in_pandas.reset_index().to_csv(csv_name, index=False)
                
                tac(_start_time)
    except:
        pass