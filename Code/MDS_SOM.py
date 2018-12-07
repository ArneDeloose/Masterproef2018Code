import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')

net, raw_data=AD.fit_SOM(list_files)

U=AD.calc_Umat(net)
import matplotlib.pyplot as plt
plt.imshow(U)


#plot MDS for full data
net_features=AD.calc_net_features(net)
D=AD.calc_dist_matrix2(raw_data, net_features, 1)
pos=AD.calc_pos(D)
AD.plot_MDS2(pos)
