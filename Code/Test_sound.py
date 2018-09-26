import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)

import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
sample_rate, samples, t, total_time,steps= AD.spect('Test.wav');


samples_dummy=samples[int(sample_rate):int(2*sample_rate)]
AD.spect_plot(samples_dummy,sample_rate)
tensor=AD.calc_tensor('temp_figure.png');
#tensor=AD.filter_noise(tensor)

path2='C:\\Users\\arne\\Documents\\School\\Thesis\\test_ROI.png'
ellipses=AD.ROI(path2, [10,5])


import pywt
original = pywt.data.camera()
fig1=AD.wave_plot(original,'bior1.3')
fig2=AD.wave_plot(tensor,'bior1.3')

#w = pywt.Wavelet('db3') #example
#print(w) #show properties


# Wavelet transform of image, and plot approximation and details


##
#for x in range(steps):
    #samples_dummy=samples[int(x*sample_rate/2):int((x+1)*sample_rate/2)]
    #AD.spect_plot(samples_dummy,sample_rate)
    #tensor=AD.calc_tensor('temp_figure.png');

    #analysis of the plot saved as 'temp_figure.png'