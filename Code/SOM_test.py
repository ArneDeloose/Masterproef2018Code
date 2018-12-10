import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

import numpy as np

#file_name1='eser-1_ppip-2µl1µA043_AEI.wav'
#file_names=('eser-1µl1µA030_ACH.wav', 'eser-1_ppip-2µl1µA043_AEI.wav')
file_names=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')
rectangles1, regions1, spectros1=AD.spect_loop(file_names[0])

freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD.loading_init()
num_reg=AD.calc_num_regions(regions1)
features1, features_key1, features_freq1=AD.calc_features(rectangles1, regions1, regions_temp, num_reg, list_bats, num_total)

#AD.write_output(file_names)

raw_data=features1
network_dim = (5,5)
n_iter = 10000
init_learning_rate = 0.01
normalise_data = False
normalise_by_column = False

net=AD.SOM(raw_data, network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column)

score_BMU=AD.calc_BMU_scores(raw_data, net)

U=AD.calc_Umat(net)
import matplotlib.pyplot as plt
plt.imshow(U)
plt.show()

#plot MDS for full data

net_features=AD.calc_net_features(net)

D=AD.calc_dist_matrix2(net_features, features1, 1)

pos=AD.calc_pos(D)
AD.plot_MDS2(pos)


