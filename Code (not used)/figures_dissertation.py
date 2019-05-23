import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD2_Spectro as AD2
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

rectangles1, regions1, spectros1=AD2.spect_loop('ppip-1µl1µB011_ABJ.wav')


import matplotlib.pyplot as plt


para_adj=AD1.set_parameters()
spec_min=para_adj[4]
spec_max=para_adj[5]
t_max=para_adj[6]

   
f, ax1 = plt.subplots()
ax1.imshow(spectros1[2], origin='lower', aspect='auto')

labels_X = [item.get_text() for item in ax1.get_xticklabels()]
labels_Y = [item.get_text() for item in ax1.get_yticklabels()]
labels_X[1]=0
labels_X[2]=20
labels_X[3]=40
labels_X[4]=60
labels_X[5]=80
labels_X[-2]=t_max
for i in range(1, len(labels_Y)-1):
    labels_Y[i]=int((spec_max-spec_min)*(i-1)/(len(labels_Y)-3)+spec_min)
ax1.set_xticklabels(labels_X)
ax1.set_yticklabels(labels_Y)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (kHz)')
plt.show()
f.savefig('spectrogram_nosub.eps', format='eps', dpi=1000)
plt.close()


#binary
import cv2
import numpy as np
import matplotlib.patches as patches

file_name1='ppip-1µl1µA044_AAT.wav' 
rectangles1, regions1, spectros1=AD2.spect_loop(file_name1)

k=33

para=AD1.set_parameters()
kern=para[1]
X=para[0]

spect_norm=spectros1[k]

ret,thresh = cv2.threshold(spect_norm, X, 256, cv2.THRESH_BINARY)
#dilation
kernel = np.ones((kern,kern), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
ctrs, _= cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

f, ax1 = plt.subplots()
ax1.imshow(img_dilation, origin='lower', aspect='auto')

labels_X = [item.get_text() for item in ax1.get_xticklabels()]
labels_Y = [item.get_text() for item in ax1.get_yticklabels()]
labels_X[1]=0
labels_X[2]=20
labels_X[3]=40
labels_X[4]=60
labels_X[5]=80
labels_X[-2]=t_max
for i in range(1, len(labels_Y)-1):
    labels_Y[i]=int((spec_max-spec_min)*(i-1)/(len(labels_Y)-3)+spec_min)
ax1.set_xticklabels(labels_X)
ax1.set_yticklabels(labels_Y)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (kHz)')
plt.show()
f.savefig('spect_binary.eps', format='eps', dpi=1000)
plt.close()

#show regions
f, ax1 = plt.subplots()
ax1.imshow(spectros1[k], origin='lower', aspect='auto')
dummy=rectangles1[k].shape
for j in range(dummy[1]):
   rect = patches.Rectangle((rectangles1[k][0,j],rectangles1[k][1,j]),
                                rectangles1[k][2,j],rectangles1[k][3,j],
                                linewidth=1,edgecolor='r',facecolor='none')
   ax1.add_patch(rect)
labels_X = [item.get_text() for item in ax1.get_xticklabels()]
labels_Y = [item.get_text() for item in ax1.get_yticklabels()]
labels_X[1]=0
labels_X[2]=20
labels_X[3]=40
labels_X[4]=60
labels_X[5]=80
labels_X[-2]=t_max
for i in range(1, len(labels_Y)-1):
    labels_Y[i]=int((spec_max-spec_min)*(i-1)/(len(labels_Y)-3)+spec_min)
ax1.set_xticklabels(labels_X)
ax1.set_yticklabels(labels_Y)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (kHz)')
plt.show()
f.savefig('ROI.eps', format='eps', dpi=1000)
plt.close()


#MDS and TSNE

#initial setting
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD4_SOM as AD4
import AD5_MDS as AD5
import matplotlib.pyplot as plt
import numpy as np

path0='C:/Users/arne/Documents/School/Thesis/Templates_explore'; #Change this to directory that stores the data
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

X_final, Y_final, net, D=AD4.evaluation_SOM(path=path0, dim1=5, dim2=5, Plot_Flag=False)

X_transform=np.matmul(D, X_final)

dist=AD5.calc_dist_matrix(X_transform, 1)
pos=AD5.calc_pos(dist)

_, _, _, _, list_bats, _, _, _, _, _=AD1.loading_init()

import matplotlib.pyplot as plt

s = 10
plot1=plt.scatter(pos[0:17, 0], pos[0:17, 1], color='turquoise', marker='o', s=s, lw=1, label='eser')
plot2=plt.scatter(pos[17:23, 0], pos[17:23, 1], color='red', marker='>', s=s, lw=1, label='mdau')
plot3=plt.scatter(pos[23:44, 0], pos[23:44, 1], color='green', marker='v', s=s, lw=1, label='nlei')
plot4=plt.scatter(pos[44:62, 0], pos[44:62, 1], color='blue', marker='^', s=s, lw=1, label='pnat')
plot5=plt.scatter(pos[62:, 0], pos[62:, 1], color='orange', marker='<', s=s, lw=1, label='ppip')
plt.legend(handles=[plot1,plot2, plot3, plot4, plot5])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.close()

