from __future__ import division #changes / to 'true division'
import scipy.io.wavfile
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
import cv2
#import AD_functions as AD
import pywt
import matplotlib.patches as patches
from skimage.measure import compare_ssim as ssim
from sklearn import manifold
import os
from scipy.cluster.hierarchy import dendrogram, linkage

path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)

path_arrays=path+'/Templates_arrays/'
list_files_arrays=os.listdir(path_arrays + 'pnat')

for i in range(len(list_files_arrays)):
    path_temp=path_arrays+ 'pnat/' + str(list_files_arrays[i])
    temp=np.load(path_temp)
    plt.imshow(temp, origin='lower')
    path_image=path+'/Templates_images/' +'pnat/' + str(list_files_arrays[i])
    path_image=path_image[:-4]
    plt.savefig(path_image)
    plt.close()

