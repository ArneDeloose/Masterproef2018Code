def calc_output(list_files, net, **optional): #Optional only works on non TE data
    #loading
    t=np.zeros((8,))
    t[0]=time.time()
    freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp=AD.loading_init(**optional)
    t[1]=time.time()
    list_files2=list_files
    list_bats, colors_bat=AD.set_batscolor()
    #create empty dictionaries
    net_label={}
    features={}
    features_key={}
    features_freq={}
    rectangles={}
    regions={}
    spectros={}
    optional['write']=True
    t[2]=time.time()
    t_temp=np.zeros((len(list_files), 5))
    #run clustering and save output    
    for i in range(len(list_files2)):
        t_temp[i, 0]=time.time()
        rectangles[i], regions[i], spectros[i]=AD.spect_loop(list_files2[i], **optional)
        t_temp[i, 1]=time.time()
        num_reg=AD.calc_num_regions(regions[i])
        t_temp[i, 2]=time.time()
        features[i], features_key[i], features_freq[i]=AD.calc_features(rectangles[i], regions[i], templates, num_reg, list_bats, num_total)
        t_temp[i, 3]=time.time()
        net_label[i]=AD.calc_BMU_scores(features[i], net)
        t_temp[i, 4]=time.time()
    t[3]=np.mean(t_temp[:,0])
    t[4]=np.mean(t_temp[:,1])
    t[5]=np.mean(t_temp[:,2])
    t[6]=np.mean(t_temp[:,3])
    t[7]=np.mean(t_temp[:,4])
    return(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, t)   

import time

import numpy as np
t=np.zeros((14,))
dt=np.zeros((13,))

t[0]=time.time()

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')
Full_flag=False

t[1]=time.time()

#option 1 (new map)
#Fit self-organising map (can take a while)
#To save the map, add argument 'export' and set it to a name (e.g. AD.fit_SOM(list_files, full=Full_flag, export='map1')
Dim1=5
Dim2=5
net, raw_data=AD.fit_SOM(list_files, full=Full_flag, dim1=Dim1, dim2=Dim2)

t[2]=time.time()

#Calculate regions (can take a while)
#M: number of regions matching per neuron
net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, t_new=calc_output(list_files, net)
t[3:11]=t_new
t[11]=time.time()
full_region, full_rectangle, full_spectro, full_name=AD.rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net, dim1=Dim1, dim2=Dim2)
t[12]=time.time()
M=AD.calc_matching(full_name, dim1=Dim1, dim2=Dim2)

t[13]=time.time()

for i in range(len(t)-1):
    dt[i]=t[i+1]-t[i]


#5 and 10: empty (due to average of loop)
import matplotlib.pyplot as plt
plt.plot(dt[0:5], 's')
plt.xticks([0, 1, 2, 3, 4], ['import', 'fit map', 'start calc_output', 'loading init', 'loading init 2'])
plt.show()
plt.close()
plt.plot(dt[6:10], 's')
plt.xticks([0, 1, 2, 3, 4], ['spect loop', 'calc num_reg', 'calc features', 'calc BMU scores'])
plt.show()
plt.close()
plt.plot(dt[11:], 's')
plt.xticks([0, 1], ['rearrange regions', 'calc M'])
plt.show()
plt.close()

    
    