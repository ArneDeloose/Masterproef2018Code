import scipy.io.wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import AD_functions as AD
import pywt
import matplotlib.patches as patches
#import matplotlib.pyplot as plt

def set_parameters():
    X=25 #Threshold for noise
    kern=[3,3] #minimum size rectangle
    #1 point= 1/3 ms
    #1 point= 375 Hz
    #kernel: 3, 3=> minimum size of ROI: roughly 1ms and 1 kHz
    return(X, kern)

def spect(file_name):
    #Reads information from audio file
    [sample_rate,samples]=scipy.io.wavfile.read(file_name, mmap=False);
    N=len(samples); #number of samples
    t=np.linspace(0,N/sample_rate, num=N); #time_array
    total_time=N/sample_rate;
    steps=math.floor(total_time)
    microsteps=math.floor(10*(total_time-steps))
    return(sample_rate, samples, t, total_time, steps, microsteps)

def spect_plot(samples, sample_rate):
    #Makes a spectrogram, data normalised to the range [0-1]
    #Change parameters of spectrogram (window, resolution)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=('hamming'), nfft=1024)
    dummy=(spectrogram-spectrogram.min())/(spectrogram.max()-spectrogram.min())
    spect_norm=np.array(np.round(dummy*256), dtype=np.uint8)
    return(spect_norm[80:214,:]) #Frequency 30-80.25 kHz

def spect_loop(file_name):
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    X, kern=set_parameters();
    sample_rate, samples, t, total_time,steps, microsteps= AD.spect(file_name);
    rectangles={};
    regions={};
    spectros={};
    dummy_per=int(0);
    for i in range(steps):
        for j in range(10):
            samples_dummy=samples[int(i*sample_rate+sample_rate*j/10):int(i*sample_rate+sample_rate*(j+1)/10)]
            spect_norm=AD.spect_plot(samples_dummy,sample_rate)
            ctrs, dummy_flag=AD.ROI(spect_norm, [3, 3], X)
            if dummy_flag:
                rectangles[i*10+j], regions[i*10+j]=AD.ROI2(ctrs, spect_norm)
            spectros[i*10+j]=spect_norm
        dummy_per=round(100*i/steps);
        print(dummy_per, ' percent complete') #Percentage completion
    for j in range(microsteps):
        samples_dummy=samples[int((i+1)*sample_rate+sample_rate*j/10):int((i+1)*sample_rate+sample_rate*(j+1)/10)]
        spect_norm=AD.spect_plot(samples_dummy,sample_rate)
        ctrs, dummy_flag=AD.ROI(spect_norm, [3, 3], X)
        if dummy_flag:
            rectangles[(i+1)*10+j], regions[(i+1)*10+j]=AD.ROI2(ctrs, spect_norm)
        spectros[(i+1)*10*j]=spect_norm
    return(rectangles, regions, spectros)
    
def ROI(spect_norm, kern, X):
    #Image_path: location of the figure
    #kern: parameters of the kernel size
    len_flag=True
    #binary
    #Conversion to uint8 for contours
    ret,thresh = cv2.threshold(spect_norm,X,256,cv2.THRESH_BINARY)
    #dilation
    kernel = np.ones((kern[0],kern[1]), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Retr_external: retrieval mode external. Only outer edges are considered,
    #Contours within other contours aren't allowed, any holes are ignored
    #Chain_approx_simple: stores only the outer points of the rectangle (4 points)   
    #set len_flag: True if there are contours, False if image is empty
    if len(ctrs)==0:
        len_flag=False #Empty image
    return(ctrs, len_flag)

#Only use if the image is not empty
def ROI2(ctrs, spect_norm):    
    Mask = np.zeros((4, len(ctrs)), dtype=np.uint8)
    regions={}
    for i, ctr in enumerate(ctrs):
        # Get bounding box
        #x, y: lower left corner
        #w, h: width and height
        x, y, w, h =cv2.boundingRect(ctr);
        Mask[0, i]=int(x);
        Mask[1, i]=int(y);
        Mask[2, i]=int(w);
        Mask[3, i]=int(h);
        regions[i]=spect_norm[x:x+w, y:y+h]
    return(Mask, regions)


def wave_plot(data, wavelet):
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

def show_region(rectangles, spectros, i):
    f, ax1 = plt.subplots()
    ax1.imshow(spectros[i])
    dummy=rectangles[i].shape
    for j in range(dummy[1]):
       rect = patches.Rectangle((rectangles[i][0,j],rectangles[i][1,j]),
                                rectangles[i][2,j],rectangles[i][3,j],
                                linewidth=1,edgecolor='r',facecolor='none')
       # Add the patch to the Axes
       ax1.add_patch(rect)
    plt.show()
    return()