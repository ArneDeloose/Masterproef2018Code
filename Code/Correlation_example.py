import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

file_name1='eser-1_ppip-2µl1µA043_AEI.wav'


rectangles1, regions1, spectros1=AD.spect_loop(file_name1)

freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD.loading_init()
num_reg=AD.calc_num_regions(regions1)

features1, features_key1, features_freq1=AD.calc_features(rectangles1, regions1, regions_temp, num_reg, list_bats, num_total)

features=np.transpose(features1)

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.DataFrame(data=features)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
plt.show()