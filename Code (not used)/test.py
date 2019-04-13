
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)


freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD1.loading_init()

import numpy as np

a_arr=np.load('C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Templates_arrays/ppip/-8889505179326917251.npy')

for i in range(1, 7):
    a_str='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Templates_rect/ppip/' +str(i) +'.npy'
    a_rec=np.load(a_str)
    print(str(i) + ': ' + str(a_rec[3]) + ' ' + str(a_rec[2]))

