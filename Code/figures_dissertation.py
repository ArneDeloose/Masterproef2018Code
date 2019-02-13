import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

rectangles1, regions1, spectros1=AD.spect_loop('ppip-1µl1µB011_ABJ.wav')


import matplotlib.pyplot as plt


para_adj=AD.adjustable_parameters()
spec_min=para_adj[1]
spec_max=para_adj[2]
t_max=para_adj[4]

   
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
rectangles1, regions1, spectros1=AD.spect_loop(file_name1)

k=33

para=AD.set_parameters()
kern=para[1]
X=para[0]

spect_norm=spectros1[k]

ret,thresh = cv2.threshold(spect_norm, X, 256, cv2.THRESH_BINARY)
#dilation
kernel = np.ones((kern,kern), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
im, ctrs, hier= cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

f, ax1 = plt.subplots()
ax1.imshow(im, origin='lower', aspect='auto')

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


