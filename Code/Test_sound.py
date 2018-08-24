import AD_functions as AD
AD.path();
sample_rate, samples, t, total_time,steps= AD.spect('Test.wav');
for x in range(steps):
    samples_dummy=samples[int(x*sample_rate/2):int((x+1)*sample_rate/2)]
    AD.spect_plot(samples_dummy,sample_rate)
    #analysis of the plot saved as 'temp_figure.png'