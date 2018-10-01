#Check biggest rectangle size with some examples
#Append all ROIs to this size (add empty space)
#Define 'distance' between pictures
#Use KNN, new data (library) can improve this procedure

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)

import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
sample_rate, samples, t, total_time,steps= AD.spect('Test.wav');
image_path='C:\\Users\\arne\\Documents\\School\\Thesis\\temp_figure.png'
rectangles=AD.spect_loop(samples, sample_rate, steps, image_path)
highlight=AD.show_last(image_path, rectangles[49])
import cv2
cv2.imshow('Test',highlight)
cv2.waitKey(0)


samples_dummy=samples[int(sample_rate+50000):int(sample_rate+80000)]
AD.spect_plot(samples_dummy,sample_rate)
tensor=AD.calc_tensor('temp_figure.png');
#tensor=AD.filter_noise(tensor)

path3='C:\\Users\\arne\\Documents\\School\\Thesis\\test_ROI.png'
path2='C:\\Users\\arne\\Documents\\School\\Thesis\\temp_figure.png'
ellipses, highlight=AD.ROI(path3, [1,1])



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