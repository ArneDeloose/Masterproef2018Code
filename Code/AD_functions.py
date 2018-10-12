import scipy.io.wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import AD_functions as AD
import pywt
import matplotlib.patches as patches
from skimage.measure import compare_ssim as ssim

def set_parameters():
    X=25 #Threshold for noise binary image
    kern=[3,3] #minimum size rectangle
    thresh=0.6 #Threshold for ssim classification
    max_roi=10 #Maximum number of regions in a single spectrogram
    #1 point= 1/3 ms
    #1 point= 375 Hz
    #kernel: 3, 3=> minimum size of ROI: roughly 1ms and 1 kHz
    return(X, kern, thresh, max_roi)

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
    dummy=np.array(np.round(dummy*256), dtype=np.uint8) #Convert to grayscale
    spectro=AD.substraction(dummy[80:214,:]) #Frequency 30-80.25 kHz
    return(spectro) 

def substraction(spect):
    dummy=np.zeros([len(spect),len(spect[0,:])])
    spectro=np.maximum(dummy, spect-spect.mean()) #Take maximum of zero and value to remove negative values
    spectro=np.array(spectro, dtype=np.uint8) #Convert back from float to uint8
    return(spectro)

def spect_loop(file_name):
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    X, kern, _, _=AD.set_parameters()
    sample_rate, samples, t, total_time,steps, microsteps= AD.spect(file_name);
    rectangles={};
    regions={};
    spectros={};
    for i in range(steps):
        for j in range(10):
            samples_dummy=samples[int(i*sample_rate+sample_rate*j/10):int(i*sample_rate+sample_rate*(j+1)/10)]
            spect_norm=AD.spect_plot(samples_dummy,sample_rate)
            ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
            if dummy_flag:
                rectangles[i*10+j], regions[i*10+j]=AD.ROI2(ctrs, spect_norm)
            spectros[i*10+j]=spect_norm
    for j in range(microsteps):
        samples_dummy=samples[int((i+1)*sample_rate+sample_rate*j/10):int((i+1)*sample_rate+sample_rate*(j+1)/10)]
        spect_norm=AD.spect_plot(samples_dummy,sample_rate)
        ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
        if dummy_flag:
            rectangles[(i+1)*10+j], regions[(i+1)*10+j]=AD.ROI2(ctrs, spect_norm)
        spectros[(i+1)*10+j]=spect_norm
    rectangles2, regions2=AD.overload(rectangles, regions)
    return(rectangles2, regions2, spectros)
    
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
    Mask = np.zeros((4, 0), dtype=np.uint8) #empty array
    regions={}
    count=0
    for i, ctr in enumerate(ctrs):
        # Get bounding box
        #x, y: lower left corner
        #w, h: width and height
        x, y, w, h =cv2.boundingRect(ctr);
        #Signal needs to be broader than 4875 Hz (roughly 5 kHz)
        #And longer than 5 ms
        if h>13 and w>3:
            temp=np.zeros((4, 1), dtype=np.uint8) #create temporary aray
            temp[0, 0]=int(x) #Fill in values
            temp[1, 0]=int(y)
            temp[2, 0]=int(w)
            temp[3, 0]=int(h)
            Mask=np.concatenate((Mask, temp), axis=1) #Append array per column
            regions[count]=spect_norm[y:y+h, x:x+w]
            count+=1
    return(Mask, regions)

def overload(rectangles, regions): #deletes entries with ten or more rectangles
    _, _, _, max_roi=set_parameters()
    rectangles2=rectangles.copy() #copy dictionaries
    regions2=regions.copy()
    for i,j in rectangles.items(): #iterate over all items
        if len(rectangles[i][0,:])>max_roi:
            rectangles2.pop(i);
            regions2.pop(i);
    return(rectangles2, regions2)

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
    plt.title(i)
    plt.show()
    return()

def show_mregions(rectangles, spectros):
    for i in range(len(rectangles)):
        AD.show_region(rectangles, spectros, i)
        input('Press enter to continue')
    return()

def compare_img(img1, img2):
    si=(len(img2[1,:]), len(img2))
    img1_new=cv2.resize(img1, si)
    score=ssim(img1_new, img2)
    return(score)

def resize_img_plot(img1,img2):
    si=(len(img2[1,:]), len(img2))
    img1_new=cv2.resize(img1, si)
    f, (ax1, ax2) = plt.subplots(2,1)
    ax1.imshow(img1)
    ax2.imshow(img1_new)
    plt.show()
    return()

def compare_img_plot(img1,img2):
    f, (ax1, ax2) = plt.subplots(2,1)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()
    return()

def create_smatrix(rectangles, spectros, num_classes): #creates empty score matrix
    x=len(spectros)
    dummy=np.zeros(x, dtype=np.uint8)
    for i,j in rectangles.items():
        dummy[i]=len(rectangles[i][0,:])
    y=dummy.max()
    s_mat=np.zeros((y, x, num_classes), dtype=np.float)
    return(s_mat)

def calc_smatrix(s_mat, regions, templates, num): #Fills s_mat
    s_mat2=s_mat.copy()
    a=len(templates)
    dummy=np.zeros(a, dtype=np.float)
    for i,d in regions.items():
        for j,d in regions[i].items():
            for k in range(a):
                dummy[k]=AD.compare_img(regions[i][j], templates[k])
            s_mat2[j, i, num]=dummy.max()
    return(s_mat2)

def create_cmatrix(rectangles, spectros): #creates empty classify matrix
    x=len(spectros)
    dummy=np.zeros(x, dtype=np.uint8)
    for i,j in rectangles.items():
        dummy[i]=len(rectangles[i][0,:])
    y=dummy.max()
    c_mat=np.zeros((y,x), dtype=np.uint8)
    return(c_mat)

def calc_cmatrix(c_mat, s_mat): #Fills c_mat
    _, _, thresh, _=set_parameters();
    c_mat2=c_mat.copy()
    y=len(c_mat) #rows
    x=len(c_mat[0]) #colums
    for i in range(x):
        for j in range(y):
            index_max=np.argmax(s_mat[j,i,:])
            value_max=s_mat[j,i,:].max()
            if value_max>thresh:
                c_mat2[j,i]=index_max+2
            else:
                if np.sum(s_mat[j,i,:], )!=0: #Signal present
                    c_mat2[j,i]=1
    return(c_mat2)

#0: empty
#1: not-classified
#n: class (n-2)

def calc_result(c_mat, num_classes):
    res=np.zeros([num_classes+2,1], dtype=np.uint8)
    for i in range(num_classes+2):
        res[i]=np.sum(c_mat==i)
    return(res)

def loop_res(rectangles, spectros, regions, templates): #result for a single class
    s_mat=AD.create_smatrix(rectangles, spectros, 1)
    s_mat=AD.calc_smatrix(s_mat, regions, templates, 0)
    c_mat=AD.create_cmatrix(rectangles, spectros)
    c_mat=AD.calc_cmatrix(c_mat, s_mat)
    res=AD.calc_result(c_mat, 1)
    return(res, c_mat, s_mat)

def create_template_set(): #temp function storing a template set
    file_name1='ppip-1µl1µA044_AAT.wav' #training set
    rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
    img1=regions1[0][0]
    img2=regions1[1][0]
    img3=regions1[2][0]
    img4=regions1[3][0]
    img5=regions1[4][0]
    img6=regions1[5][0]
    img7=regions1[6][0]
    img8=regions1[7][0]
    img9=regions1[8][0]
    img10=regions1[9][0]
    img11=regions1[11][0]
    img12=regions1[12][0]
    img13=regions1[13][0]
    img14=regions1[14][0]
    img15=regions1[15][0]
    img16=regions1[16][0]
    img17=regions1[17][0]
    img18=regions1[18][0]
    img19=regions1[20][0]
    img20=regions1[21][0]
    templates_0={0: img1, 1: img2, 2: img3, 3: img4,
             4: img5, 5: img6, 6: img7, 7: img8,
             8: img9, 9: img10, 10: img11, 11: img12,
             12: img13, 13: img14, 14: img15, 15: img16,
             16: img17, 17: img18, 18: img19, 19: img20}
    return(templates_0)

def show_class(class_num, c_mat, rectangles, regions, spectros):
    for i in range(len(c_mat)): #Rows, region
        for j in range(len(c_mat[0,:])): #Colums, time
            if c_mat[i,j]==class_num:
                f, ax1 = plt.subplots()
                ax1.imshow(spectros[j])
                rect = patches.Rectangle((rectangles[j][0, i],rectangles[j][1, i]),
                                rectangles[j][2, i],rectangles[j][3, i],
                                linewidth=1,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                ax1.add_patch(rect)
                plt.title(j)
                plt.show()
                input('Press enter to continue')
    return()
