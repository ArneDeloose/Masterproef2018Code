#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD3_Features as AD3 
import AD4_SOM as AD4 

path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
import numpy as np
import matplotlib.pyplot as plt

freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD1.loading_init()

#set variables
rectangles1=rectangles_temp
regions1=regions_temp

templates1=regions_temp

#calc features
features=AD3.calc_features2(rectangles1, regions1, templates1, list_bats, num_total)
X=np.transpose(features)

Y=np.zeros((75,), dtype=np.uint8)


#fill in Y matrix
Y[0:16]=0
Y[17:23]=1
Y[23:29]=2
Y[29:47]=3
Y[47:75]=4


Full_flag=False
list_files=['eser-1_ppip-2µl1µA043_AEI.WAV', 'eser-1_ppip-2µl1µA048_AFT.WAV',
            'ppip-1µl1µB011_ABJ.WAV', 'ppip-1µl1µA045_AAS.WAV', 'mdau-1µl1µA052_AJP.WAV']
#adaptable code

Dim1=100
Dim2=100
D=np.load(path+'/D1.npy')

#raw_data2=np.concatenate((raw_data, features), axis=1)

net, raw_data=AD4.fit_SOM(list_files, full=Full_flag, dim1=Dim1, dim2=Dim2, DML=D)
#, features=features

#plot net
m=features.shape[0]

count=np.zeros((6,), dtype=np.uint8)
for i in range(features.shape[1]):
    t=features[:,i].reshape(np.array([m, 1])) 
    bmu, bmu_idx=AD4.find_bmu(t, net, m, D)
    if Y[i]==0:
        col='ro'
        if count[0]==0:
            label='eser'
            count[0]+=1
        else:
            label=None
    elif Y[i]==1:
        col='g-'
        if count[1]==0:
            label='mdau'
            count[1]+=1
        else:
            label=None
    elif Y[i]==2:
        col='b*'
        if count[2]==0:
            label='nlei'
            count[2]+=1
        else:
            label=None
    elif Y[i]==3:
        col='cv'
        if count[3]==0:
            label='pnat'
            count[3]+=1
        else:
            label=None
    elif Y[i]==4:
        col='m^'
        if count[4]==0:
            label='ppip'
            count[4]+=1
        else:
            label=None
    else:
        col='k,'
        if count[5]==0:
            label='other'
            count[5]+=1
        else:
            label=None
    plt.plot(bmu_idx[0], bmu_idx[1], col, label=label)

plt.title('SOM, different data, D-matrix, 100 X 100')
plt.legend()
plt.savefig('SOM_Jeffrey_2')

   
