import cv2
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import time
import haralick_features as hf
import sys

def getDictMatrices(matrix, window_size):
    row_first = 0
    row_last = window_size
    center = int((window_size-1)/2)
    dict_of_matrices = []
    keys = []

    for i in range(matrix.shape[0] + 1):
        
        if row_last == matrix.shape[0] + 1:
            break
        
        column_first = 0
        column_last = window_size
        for j in range(matrix.shape[1] + 1):
            
            if column_last == matrix.shape[1] + 1:
                break
            
            window = matrix[row_first:row_last, column_first:column_last]#.copy()
            key_name = 'cell_%s_%s'%(row_first + center, column_first + center)
            dict_of_matrices.append(window)
            keys.append(key_name)
            column_first+=1
            column_last+=1

        row_first+=1
        row_last+=1
    
    return dict_of_matrices, keys

"""
def processMatrix(matrix, orientation):
    orientation_dict = {'horizontal': hf.horizontalGTSDM(matrix, 256),
                        'vertical': hf.verticalGTSDM(matrix, 256),
                        'l_diagonal': hf.l_diagonalGTSDM(matrix, 256),
                        'r_diagonal': hf.r_diagonalGTSDM(matrix, 256)}
    
    image = orientation_dict['orientation']
    
    return [hf.ASM(image), 
            hf.constrast(image), 
            hf.getCorrelation(image),
            hf.varianceMatrix(image),
            hf.inverseDifferenceMomentMatrix(matrix),
            hf.sumAverageMatrix(matrix),
            hf.sumVarianceMatrix(matrix),
            hf.sumEntropyMatrix(matrix),
            hf.entropyMatrix(matrix),
            hf.differenceVariance(matrix),
            hf.differenceEntropy(matrix),
            hf.infoCorrelation1(matrix),
            hf.infoCorrelation2(matrix),]
"""

def processMatrix_horizontal(matrix):
    image = hf.horizontalGTSDM(matrix, 256)

    return [hf.ASM(image), 
            hf.constrastMatrix(image), 
            hf.getCorrelation(image),
            hf.varianceMatrix(image),
            hf.inverseDifferenceMomentMatrix(matrix),
            hf.sumAverageMatrix(matrix),
            hf.sumVarianceMatrix(matrix, hf.sumAverageMatrix(matrix)),
            hf.sumEntropyMatrix(matrix),
            hf.entropyMatrix(matrix),
            hf.differenceVariance(matrix),
            hf.differenceEntropy(matrix),
            hf.infoCorrelation1(matrix, hf.entropyMatrix(matrix)),
            hf.infoCorrelation2(matrix, hf.entropyMatrix(matrix)),]

def processMatrix_vertical(matrix):
    image = hf.verticalGTSDM(matrix, 256)

    return [hf.ASM(image), 
            hf.constrastMatrix(image), 
            hf.getCorrelation(image),
            hf.varianceMatrix(image),
            hf.inverseDifferenceMomentMatrix(matrix),
            hf.sumAverageMatrix(matrix),
            hf.sumVarianceMatrix(matrix, hf.sumAverageMatrix(matrix)),
            hf.sumEntropyMatrix(matrix),
            hf.entropyMatrix(matrix),
            hf.differenceVariance(matrix),
            hf.differenceEntropy(matrix),
            hf.infoCorrelation1(matrix, hf.entropyMatrix(matrix)),
            hf.infoCorrelation2(matrix, hf.entropyMatrix(matrix)),]

def processMatrix_l_diagonal(matrix):
    image = hf.l_diagonalGTSDM(matrix, 256)

    return [hf.ASM(image), 
            hf.constrastMatrix(image), 
            hf.getCorrelation(image),
            hf.varianceMatrix(image),
            hf.inverseDifferenceMomentMatrix(matrix),
            hf.sumAverageMatrix(matrix),
            hf.sumVarianceMatrix(matrix, hf.sumAverageMatrix(matrix)),
            hf.sumEntropyMatrix(matrix),
            hf.entropyMatrix(matrix),
            hf.differenceVariance(matrix),
            hf.differenceEntropy(matrix),
            hf.infoCorrelation1(matrix, hf.entropyMatrix(matrix)),
            hf.infoCorrelation2(matrix, hf.entropyMatrix(matrix)),]

def processMatrix_r_diagonal(matrix):
    image = hf.r_diagonalGTSDM(matrix, 256)

    return [hf.ASM(image), 
            hf.constrastMatrix(image), 
            hf.getCorrelation(image),
            hf.varianceMatrix(image),
            hf.inverseDifferenceMomentMatrix(matrix),
            hf.sumAverageMatrix(matrix),
            hf.sumVarianceMatrix(matrix, hf.sumAverageMatrix(matrix)),
            hf.sumEntropyMatrix(matrix),
            hf.entropyMatrix(matrix),
            hf.differenceVariance(matrix),
            hf.differenceEntropy(matrix),
            hf.infoCorrelation1(matrix, hf.entropyMatrix(matrix)),
            hf.infoCorrelation2(matrix, hf.entropyMatrix(matrix)),]


def tac(_start_time):
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {} hour - {} min - {} sec'.format(t_hour,t_min,t_sec))

images_path = '/home/nicolas/Documents/tesis/pca_images_1/'
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
        #blue = blue.copy()[0:1000, 0:1000]
        #green = green.copy()[0:1000, 0:1000]
        #red = red.copy()[0:1000, 0:1000]
        chanels = { 'blue': blue,
                    'green': green,
                    'red': red}
        for chanel in chanels:
            matrix = chanels[chanel]
            list_of_matrices, keys = getDictMatrices(matrix, int(sys.argv[ 1 ]))
            print(len(list_of_matrices))
            for method in spatial_dependence_matrix:
                print(method)
                _start_time = time.time()
                
                pool = Pool(num_cores)
                result = pool.map(func=spatial_dependence_matrix[method], iterable = list_of_matrices,)
                data_in_dict = {keys[i]: result[i] for i in range(len(keys))}
                data_in_pandas = pd.DataFrame.from_dict(data_in_dict, orient='index', 
                                                        columns=['ASM', 
                                                                'contrast', 
                                                                'correlation',
                                                                'variance',
                                                                'inverseDifferenceMoment',
                                                                'sumAverage',
                                                                'sumVariance',
                                                                'sumEntropy',
                                                                'entropy',
                                                                'differenceVariance',
                                                                'differenceEntropy',
                                                                'infoCorrelation1',
                                                                'infoCorrelation2'])
                pool.terminate()
                
                os.chdir(csv_path)
                csv_name = '%s_%s_%s.csv'%(image, chanel, method)
                data_in_pandas.reset_index().to_csv(csv_name, index=False)
                
                tac(_start_time)
    except:
        pass