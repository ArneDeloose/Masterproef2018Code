
import numpy as np

def calc_dist_matrix2(net_features, axis, **optional): #calculates distance per column (if axis=1)
    if 'raw_data' in optional:
        array=np.concatenate((net_features, optional['raw_data']), axis=1)
    else:
        array=net_features
    D=np.zeros((array.shape[axis], array.shape[axis]), dtype=np.float)
    for i in range(array.shape[axis]):
        for j in range(array.shape[axis]):
            D[i,j]=sum((array[:, i]-array[:,j])**2)
    return(D)