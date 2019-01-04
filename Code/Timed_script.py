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
    t[3]=np.mean(t_temp[:, 0])
    t[4]=np.mean(t_temp[:, 1])
    t[5]=np.mean(t_temp[:, 2])
    t[6]=np.mean(t_temp[:, 3])
    t[7]=np.mean(t_temp[:, 4])
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

##calc features

import math

def calc_features(rectangles, regions, templates, num_reg, list_bats, num_total):
    t=np.zeros((14,))
    features=np.zeros((len(templates)+7, num_reg))
    features_freq=np.zeros((7, num_reg)) #unscaled freq info
    count=0
    features_key={}
    for i,d in regions.items():
        for j,d in regions[i].items():
            t=np.hstack([t, np.zeros((14,1))])
            t[0, count]=time.time()
            features_key[count]=(i,j)
            features[0, count]=rectangles[i][3,j] #freq range
            features[1, count]=rectangles[i][1,j] #min freq
            features[2, count]=rectangles[i][1,j]+rectangles[i][3,j] #max freq
            features[3, count]=rectangles[i][1,j]+rectangles[i][3,j]/2 #av freq
            features[4, count]=rectangles[i][2,j] #duration
            t[1, count]=time.time()
            index=np.argmax(regions[i][j]) #position peak frequency
            l=len(regions[i][j][0,:]) #number of timesteps
            a=index%l #timestep at peak freq
            b=math.floor(index/l) #frequency at peak freq
            features[5, count]=a/l #peak frequency T
            features[6, count]=b+rectangles[i][1,j] #peak frequency F
            t[2, count]=time.time()
            for k in range(len(templates)):
                features[k+7, count]=AD.compare_img2(regions[i][j], templates[k])
            features_freq[:, count]=features[:7, count]
            t[3, count]=time.time()
            count+=1
    #Feature scaling, half of the clustering is based on freq and time information
    for k in range(7):
        features[k,:]=(num_total/7)*(features[k, :]-features[k, :].min())/(features[k, :].max()-features[k, :].min())
    return(features, features_key, features_freq, t)

t2=np.zeros((14,))
dt2=np.zeros((13,))

freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp=AD.loading_init()
list_bats, colors_bat=AD.set_batscolor()
net_label={}
features={}
features_key={}
features_freq={}
rectangles={}
regions={}
spectros={}
t_temp=np.zeros((len(list_files), 5))
#run clustering and save output    
for i in range(len(list_files)):
    rectangles[i], regions[i], spectros[i]=AD.spect_loop(list_files2[i], write=True)
    num_reg=AD.calc_num_regions(regions[i])
    features[i], features_key[i], features_freq[i], t_temp=calc_features(rectangles[i], regions[i], templates, num_reg, list_bats, num_total)

for i in range(14):
    t2[i]=np.mean(t_temp[i, :])

for i in range(len(t2)-1):
    dt2[i]=t2[i+1]-t2[i]

    