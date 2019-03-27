import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')

net, raw_data=AD.fit_SOM(list_files)

net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2=AD.calc_output(list_files, net)

full_region, full_rectangle, full_spectro, full_name=AD.rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net)


import matplotlib.pyplot as plt

dim1=0
dim2=0
point=0
freq1_index=50

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
FI_matrix=AD.calc_FI_matrix(full_region[dim1][dim2][point])
ax3.imshow(FI_matrix, origin='lower', aspect='auto')
plt.draw()
    
labels_X = [item.get_text() for item in ax3.get_xticklabels()] #original labels
labels_x=list() #new labels
labels_x.append(labels_X[0])
for i in range(1, len(labels_X)):
    labels_x.append(str(float((float(labels_X[i])+freq1_index)*0.375)))
    
ax3.set_xticklabels(labels_x)
ax3.set_xlabel('Frequency (kHz)')
ax3.set_ylabel('Intensity')
