#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
import numpy as np
import dml


#read in data
freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD.loading_init()

#set variables
rectangles1=rectangles_temp
regions1=regions_temp
templates1=regions_temp

#calc features
features=AD.calc_features2(rectangles1, regions1, templates1, list_bats, num_total)
X=np.transpose(features)

Y=np.zeros((85,), dtype=np.uint8)


#fill in Y matrix
Y[0:16]=0
Y[17:23]=1
Y[23:29]=2
Y[29:47]=3
Y[47:85]=4

#...

#apply 

model=dml.anmm.ANMM()
model.fit(X,Y)
A=model.transformer()

#Export A-matrix
np.save(path + '/' + 'DML_matrix' + '.npy', A)

