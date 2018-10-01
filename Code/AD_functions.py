def spect(name):
    import scipy.io.wavfile
    import numpy as np
    import math
    [sample_rate,samples]=scipy.io.wavfile.read(name, mmap=False);
    N=len(samples); #number of samples
    t=np.linspace(0,N/sample_rate, num=N); #time_array
    total_time=N/sample_rate;
    steps=math.floor(total_time)
    return(sample_rate, samples, t, total_time, steps)

def spect_plot(samples, sample_rate):
    import matplotlib.pyplot as plt
    from scipy import signal
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram, cmap='Greys')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    plt.ylim(10000,80000) #normal values: 10-80k
    plt.axis('off')
    plt.savefig('temp_figure.png')
    return()

def spect_loop(samples, sample_rate, steps, path):
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    import AD_functions as AD
    rectangles={};
    dummy_per=int(0);
    for i in range(steps):
        for j in range(10):
            samples_dummy=samples[int(i*sample_rate+sample_rate*j/10):int(i*sample_rate+sample_rate*(j+1)/10)]
            AD.spect_plot(samples_dummy,sample_rate)
            ctrs, dummy_flag=AD.ROI(path, [1, 1])
            if dummy_flag:
                rectangles[i*10+j]=AD.ROI2(ctrs)
        dummy_per=round(100*i/steps);
        print(dummy_per, ' percent complete') #Percentage completion
    return(rectangles)

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
    
def ROI(image_path, kern):
    #Image_path: location of the figure
    #kern: parameters of the kernel size
    len_flag=True
    import cv2
    import numpy as np
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #dilation
    kernel = np.ones((kern[0],kern[1]), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #set len_flag: True if there are contours, False if image is empty
    if len(ctrs)==0:
        len_flag=False #Empty image
    return(ctrs, len_flag)

#Only use if the image is not empty
def ROI2(ctrs):    
    import numpy as np
    import cv2
    Mask = np.zeros((4, len(ctrs)), dtype=float)
    for i, ctr in enumerate(ctrs):
        # Get bounding box
        #x, y: lower left corner
        #w, h: width and height
        x, y, w, h =cv2.boundingRect(ctr);
        Mask[0, i]=round(x);
        Mask[1, i]=round(y);
        Mask[2, i]=round(w);
        Mask[3, i]=round(h);
    return(Mask)

def ROI_highlight(gray, ellipseMask, i): #only works for ellipses
    import cv2
    highlight= cv2.ellipse(gray, (int(ellipseMask[0,i]), int(ellipseMask[1,i])),
                               (int(ellipseMask[2,i]/2), int(ellipseMask[3,i]/2)), int(ellipseMask[4,i]),
                               int(0), int(360), int(1), int(2))
    return(highlight)

def show_last(image_path, coord):
    import cv2
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    highlight=cv2.rectangle(gray, (coord[0],coord[1]),( coord[0] + coord[2], 
                            coord[1] + coord[3] ),(0,255,0),2)
    return(highlight)