import numpy as np

y = [1,5,8,9,4,6,8,2,1]
y_pred = [1.2,5.3,7.5,8,3.6,6.3,7.9,1,1]

def evaluate(y, y_pred):
    y = np.matrix(y)
    y_pred = np.matrix(y_pred)
    error = y - y_pred
    
    return np.dot(error, error.transpose())

evaluate(y, y_pred)