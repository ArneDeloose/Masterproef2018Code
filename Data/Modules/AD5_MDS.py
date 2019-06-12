#Module for the making of MDS and TSNE plots

#Load packages
from __future__ import division #changes / to 'true division'
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

#Used to make an MDS plot of a SOM
#If 'raw_data' is given, concatenates both matrices 
#calculates distance per column (if axis=1)
def calc_dist_matrix(net_features, Axis, **optional): 
    if 'raw_data' in optional:
        array=np.concatenate((net_features, optional['raw_data']), axis=Axis)
    else:
        array=net_features
    D=np.zeros((array.shape[Axis], array.shape[Axis]), dtype=np.float)
    for i in range(array.shape[Axis]):
        for j in range(array.shape[Axis]):
            D[i,j]=sum((array[:, i]-array[:,j])**2)
    return(D)

#Calculates positions on a 2D plot based on a distance matrix
def calc_pos(dist_mat):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(dist_mat).embedding_
    return(pos)

#Calculates positions on a 2D plot based on a distance matrix
def calc_pos_TSNE(dist_mat):
    seed = np.random.RandomState(seed=3)
    tsne = manifold.TSNE(n_components=2, n_iter=3000, min_grad_norm=1e-9, random_state=seed,
                   metric="precomputed")
    pos = tsne.fit(dist_mat).embedding_
    return(pos)

#Plot an MDS with labels for each bat 
def plot_MDS(pos):
    s = 10
    plot1=plt.scatter(pos[0:38, 0], pos[0:38, 1], color='turquoise', s=s, lw=0, label='ppip')
    plot2=plt.scatter(pos[39:55, 0], pos[39:55, 1], color='red', s=s, lw=0, label='eser')
    plot3=plt.scatter(pos[56:61, 0], pos[56:61, 1], color='green', s=s, lw=0, label='mdau')
    plot4=plt.scatter(pos[62:79, 0], pos[62:79, 1], color='blue', s=s, lw=0, label='pnat')
    plot5=plt.scatter(pos[80:85, 0], pos[80:85, 1], color='orange', s=s, lw=0, label='nlei')
    #plot6=plt.scatter(pos[86:95, 0], pos[86:95, 1], color='black', s=s, lw=0, label='noise')
    plt.legend(handles=[plot1,plot2, plot3, plot4, plot5])
    plt.show()
    return()

#Plot an MDS of neurons and raw_data
def plot_MDS2(pos, dim1, dim2):
    s=10
    neur=int(dim1*dim2)
    plot1=plt.scatter(pos[0:neur, 0], pos[0:neur, 1], color='red', s=s, lw=0, label='neurons')
    plot2=plt.scatter(pos[neur+1:, 0], pos[neur+1:, 1], color='black', s=s, lw=0, label='data')
    plt.legend(handles=[plot1,plot2])
    plt.show()
    return()

#conc_features: changes features into a single array for use in an MDS
def conc_features(features):
    conc_features=np.zeros((features[0].shape[0], 0))
    for i in range(len(features)):
        conc_features=np.concatenate((conc_features, features[i]), axis=1)
    return(conc_features)
