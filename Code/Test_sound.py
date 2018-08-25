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
tensor=AD.filter_noise(tensor)

##
for x in range(steps):
    samples_dummy=samples[int(x*sample_rate/2):int((x+1)*sample_rate/2)]
    AD.spect_plot(samples_dummy,sample_rate)
    tensor=AD.calc_tensor('temp_figure.png');

    #analysis of the plot saved as 'temp_figure.png'