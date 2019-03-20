#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
import numpy as np


#read in data
freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD.loading_init()

#set variables
rectangles1=rectangles_temp
regions1=regions_temp
templates1=regions_temp

#calc features
features=AD.calc_features2(rectangles1, regions1, templates1, list_bats, num_total)
X=np.transpose(features)
Y=np.zeros((85,0), dtype=np.uint8)

#fill in Y matrix
#...

#apply 

