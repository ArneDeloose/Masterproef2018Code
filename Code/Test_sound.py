import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)

import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)


sample_rate, samples, t, total_time,steps= AD.spect('Test.wav');
samples_dummy=samples[int(1.78*sample_rate):int(1.8*sample_rate)]
AD.spect_plot(samples_dummy,sample_rate)
#tensor=AD.calc_tensor('temp_figure.png');
#tensor=AD.filter_noise(tensor)
path2='C:\\Users\\arne\\Documents\\School\\Thesis\\temp_figure.png'
ellipses, highlight=AD.ROI(path2, [1,1])

sample_rate, samples, t, total_time,steps= AD.spect('nnoc-1_ppip-1µl1µB012_AAQ.wav');
samples_dummy=samples[int(1.75*sample_rate):int(1.8*sample_rate)]
AD.spect_plot(samples_dummy,sample_rate)
path2='C:\\Users\\arne\\Documents\\School\\Thesis\\temp_figure.png'
ellipses, highlight=AD.ROI(path2, [1,1])


#path3='C:\\Users\\arne\\Documents\\School\\Thesis\\test_ROI.png'
#image2 = cv2.imread(path3)


#Example of a wavelet image
#import pywt
#original = pywt.data.camera()
#fig1=AD.wave_plot(original,'bior1.3')
#fig2=AD.wave_plot(tensor,'bior1.3')

#w = pywt.Wavelet('db3') #example
#print(w) #show properties


# Wavelet transform of image, and plot approximation and details


##
#for x in range(steps):
    #samples_dummy=samples[int(x*sample_rate/2):int((x+1)*sample_rate/2)]
    #AD.spect_plot(samples_dummy,sample_rate)
    #tensor=AD.calc_tensor('temp_figure.png');

    #analysis of the plot saved as 'temp_figure.png'