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
    plt.pcolormesh(times, frequencies, spectrogram, cmap='Greys')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    plt.ylim(10000,80000)
    plt.axis('off')
    plt.savefig('temp_figure.png')
    return()

def calc_tensor(name):
    from PIL import Image as im
    import numpy as np
    img = im.open(name).convert('LA'); #convert to grayscale
    tensor = np.asarray( img, dtype="int32" )
    tensor=tensor[:,:,0]; #We only need the first page
    return(tensor)

def filter_noise(tensor):
    a=tensor.min();
    b=int((0.1*(255-a)))
    tensor[tensor > 255-b] = 255
    return(tensor)

def wave_plot(data, wavelet):
    import pywt
    import matplotlib.pyplot as plt
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(data, 'db4')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
    return(fig)
    