def path():
    import os
    path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
    os.chdir(path)
    return()

def spect(name):
    import scipy.io.wavfile
    import numpy as np
    import math
    [sample_rate,samples]=scipy.io.wavfile.read(name, mmap=False);
    N=len(samples); #number of samples
    t=np.linspace(0,N/sample_rate, num=N); #time_array
    total_time=N/sample_rate;
    steps=2*math.floor(total_time)
    return(sample_rate, samples, t, total_time, steps)

def spect_plot(samples, sample_rate):
    import matplotlib.pyplot as plt
    from scipy import signal
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    plt.ylim(10000,80000)
    plt.savefig('temp_figure.png')
    return()