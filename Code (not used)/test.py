
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD2_Spectro as AD2
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)


freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD1.loading_init()



a_arr=np.load('C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Templates_arrays/eser/-191291779997780151.npy')
a_rec=np.load('C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Templates_rect/eser/1.npy')
