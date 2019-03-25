#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
import numpy as np
import matplotlib.pyplot as plt

freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD.loading_init()

#set variables
rectangles1=rectangles_temp
regions1=regions_temp

templates1=regions_temp

#calc features
features=AD.calc_features2(rectangles1, regions1, templates1, list_bats, num_total)
X=np.transpose(features)

Y=np.zeros((75,), dtype=np.uint8)


#fill in Y matrix
Y[0:16]=0
Y[17:23]=1
Y[23:29]=2
Y[29:47]=3
Y[47:75]=4


Full_flag=False
list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')
#adaptable code

Dim1=20
Dim2=20
D=np.load(path+'/DML_matrix.npy')

raw_data2=np.concatenate((raw_data, features), axis=1)

net, raw_data=AD.fit_SOM(list_files, full=Full_flag, dim1=Dim1, dim2=Dim2, DML=D, features=raw_data2)
#, features=features

#plot net
m=features.shape[0]

count=np.zeros((6,), dtype=np.uint8)
for i in range(features.shape[1]):
    t=features[:,i].reshape(np.array([m, 1])) 
    bmu, bmu_idx=AD.find_bmu(t, net, m, D)
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

plt.title('SOM, different data + labeled data, D-matrix, 20 X 20')
plt.legend()
plt.savefig('SOM_5')

   
