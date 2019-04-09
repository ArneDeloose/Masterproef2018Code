#Module which creates spectrograms and extracts regions from spectrograms
#There is also code to save a region as a template

#Load packages
from __future__ import division #changes / to 'true division'
import scipy.io.wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import pywt
import matplotlib.patches as patches

#load modules
import AD1_Loading as AD1
import AD2_Spectro as AD2

#Creates the parameters for a spectrogram
def spect(file_name, **optional):
    para=AD1.set_parameters()
    if 'spect_window' in optional:
        spect_window=optional['spect_window']
    else:
        spect_window=para[6]
    if 'spect_window_overlap' in optional:
        spect_window_overlap=optional['spect_window_overlap']
    else:
        spect_window_overlap=para[7]
    #Reads information from audio file
    [sample_rate,samples]=scipy.io.wavfile.read(file_name, mmap=False);
    if 'channel' in optional:
        if optional['channel']=='l':
            samples=samples[:,0]
        elif optional['channel']=='r':
            samples=samples[:,1]
    N=len(samples) #number of samples
    t=np.linspace(0,N/sample_rate, num=N); #time_array
    total_time=N/sample_rate;
    steps=math.floor((1000*total_time)/(spect_window-spect_window_overlap)) #number of windows
    return(sample_rate, samples, t, total_time, steps)

#Creates a spectrogram
def spect_plot(samples, sample_rate, **optional):
    #Makes a spectrogram, data normalised to the range [0-1]
    #Change parameters of spectrogram (window, resolution)
    para=AD1.set_parameters()
    if 'min_spec_freq' in optional:
        min_spec_freq=optional['min_spec_freq']
    else:
        min_spec_freq=para[4]
    if 'max_spec_freq' in optional:
        max_spec_freq=optional['max_spec_freq']
    else:
        max_spec_freq=para[5]
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=('hamming'), nfft=1024)
    dummy=(spectrogram-spectrogram.min())/(spectrogram.max()-spectrogram.min())
    dummy=np.array(np.round(dummy*256), dtype=np.uint8) #Convert to grayscale
    if 'nosub' in optional: #no subtraction
        spectro=dummy[min_spec_freq:max_spec_freq,:]
    else: #subtraction
        spectro=AD2.substraction(dummy[min_spec_freq:max_spec_freq,:])
    return(spectro) 

#Subtract average value of all pixels in a spectrogram to reduce noise
def substraction(spect):
    dummy=np.zeros([len(spect),len(spect[0,:])])
    spectro=np.maximum(dummy, spect-spect.mean()) #Take maximum of zero and value to remove negative values
    spectro=np.array(spectro, dtype=np.uint8) #Convert back from float to uint8
    return(spectro)

#Creates three dictionaries: rectangle with the coordinates of the regions,
#regions with the actual regions and spectros with the full spectrograms
#Empty images are skipped for regions and rectangles
def spect_loop(file_name, **optional): 
    para=AD1.set_parameters()
    if 'X' in optional:
        X=optional['X']
    else:
        X=para[0]
    if 'kern' in optional:
        kern=optional['kern']
    else:
        kern=para[1]
    if 'spect_window' in optional:
        spect_window=optional['spect_window']
    else:
        spect_window=para[6]
    if 'spect_window_overlap' in optional:
        spect_window_overlap=optional['spect_window_overlap']
    else:
        spect_window_overlap=para[7]
    #change number of steps so it matches the window and overlap
    #change algorithm so the spectrograms aren't fixed at 100 ms, but rather at the number of actual steps
    #templates are already defined so the algorthitm becomes obsolete
    rectangles={};
    regions={};
    spectros={};
    sample_rate, samples, _, _,steps= AD2.spect('Audio_data/'+ file_name, **optional);
    if 'channel' in optional:
        if 'exp_factor' in optional:
            sample_rate=optional['exp_factor']*sample_rate
            steps=math.floor(steps/optional['exp_factor'])
        else: #default value
            sample_rate=10*sample_rate
            steps=math.floor(steps/10)
    start_index=0
    stop_index=int(spect_window_overlap*len(samples)/(steps*spect_window))
    for i in range(steps):
        samples_dummy=samples[start_index:stop_index]
        start_index=stop_index
        stop_index+=int(spect_window_overlap*len(samples)/(steps*spect_window))
        spectros[i]=AD2.spect_plot(samples_dummy, sample_rate, **optional)
        ctrs, dummy_flag=AD2.ROI(spectros[i], kern, X)
        if dummy_flag:
            rectangles[i], regions[i]=AD2.ROI2(ctrs, spectros[i])
            rectangles, regions=AD2.check_overlap(rectangles, regions, spectros, i, spect_window, spect_window_overlap)
    rectangles2, regions2=AD2.overload(rectangles, regions, **optional)
    regions3=AD2.rescale_region(regions2)
    return(rectangles2, regions3, spectros)

#Checks if the same region is present twice due to overlap in two spectrograms
#If this is the case, the second entry is deleted
def check_overlap(rectangles, regions, spectros, i, spect_window, spect_window_overlap):
    overlap=spect_window_overlap/spect_window
    delta_x=int(len(spectros[0][0,:])*overlap)
    rectangles1=rectangles.copy() #replace things in copies
    regions1=regions.copy()
    if i>0: #checks if the region already exists
        if i in rectangles.keys() and i-1 in rectangles.keys(): #check if both regions actually exists
            for j in range(len(regions[i])):
                for k in range(len(regions[i-1])):
                    if rectangles[i][0,j]==rectangles[i-1][0,k]+delta_x and rectangles[i][1,j]==rectangles[i-1][1,k]: #x and y coordinates match
                        regions1[i].pop(j) #delete entry
                        rectangles1[i]=np.delete(rectangles[i], j, axis=1) #delete column
    return(rectangles1, regions1)

#Rescales the region to the range 0-255
def rescale_region(reg):
    region={}
    for i,d in reg.items():
        region[i]={}
        for j,d in reg[i].items():
            dummy=(reg[i][j]-reg[i][j].min())/(reg[i][j].max()-reg[i][j].min()) 
            region[i][j]=np.array(np.round(dummy*256), dtype=np.uint8) #Convert to grayscale
    return(region)

#Extracts regions from a spectrogram
def ROI(spect_norm, kern, X):
    #kern: parameters of the kernel size
    len_flag=True
    #binary
    #Conversion to uint8 for contours
    ret,thresh = cv2.threshold(spect_norm, X, 256, cv2.THRESH_BINARY)
    #dilation
    kernel = np.ones((kern,kern), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    _,ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    #Retr_external: retrieval mode external. Only outer edges are considered,
    #Contours within other contours aren't allowed, any holes are ignored
    #Chain_approx_simple: stores only the outer points of the rectangle (4 points)   
    #set len_flag: True if there are contours, False if image is empty
    if len(ctrs)==0:
        len_flag=False #Empty image
    return(ctrs, len_flag)

#Called in ROI. Saves the extracted regions
#This code only fires if the image is not empty (to avoid errors)
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
        if h>6 and w>3 and w<43: #2.25 kHz and length between 2 ms and 25 ms
            temp=np.zeros((4, 1), dtype=np.uint8) #create temporary aray
            temp[0, 0]=int(x) #Fill in values
            temp[1, 0]=int(y)
            temp[2, 0]=int(w)
            temp[3, 0]=int(h)
            Mask=np.concatenate((Mask, temp), axis=1) #Append array per column
            regions[count]=spect_norm[y:y+h, x:x+w]
            count+=1
    return(Mask, regions)

#Deletes all regions in a spectrogram if they exceed the max_roi parameter (to avoid very noisy spectrograms)
def overload(rectangles, regions, **optional):
    para=AD1.set_parameters()
    if 'max_roi' in optional:
        max_roi=optional['max_roi']
    else:
        max_roi=para[2]
    rectangles2=rectangles.copy() #copy dictionaries
    regions2=regions.copy()
    for i,j in rectangles.items(): #iterate over all items
        if len(rectangles[i][0,:])>max_roi:
            rectangles2.pop(i);
            regions2.pop(i);
    return(rectangles2, regions2)

#Plots a region as a red rectangle on a spectrogram
def show_region(rectangles, spectros, i, **optional):
    para=AD1.set_parameters()
    if 'spec_min' in optional:
        spec_min=optional['spec_min']
    else:
        spec_min=para[17]
    if 'spec_max' in optional:
        spec_max=optional['spec_max']
    else:
        spec_max=para[18]
    if 't_max' in optional:
        t_max=optional['t_max']
    else:
        t_max=para[6]
    f, ax1 = plt.subplots()
    ax1.imshow(spectros[i], origin='lower', aspect='auto')
    dummy=rectangles[i].shape
    for j in range(dummy[1]):
        rect = patches.Rectangle((rectangles[i][0,j],rectangles[i][1,j]),
                                rectangles[i][2,j],rectangles[i][3,j],
                                linewidth=1,edgecolor='r',facecolor='none')
        #Add the patch to the Axes
        ax1.add_patch(rect)
    plt.title(i)
    labels_X = [item.get_text() for item in ax1.get_xticklabels()]
    labels_Y = [item.get_text() for item in ax1.get_yticklabels()]
    labels_X[1]=0
    labels_X[2]=t_max
    for i in range(1, len(labels_Y)-1):
       labels_Y[i]=int((spec_max-spec_min)*(i-1)/(len(labels_Y)-3)+spec_min)
    ax1.set_xticklabels(labels_X)
    ax1.set_yticklabels(labels_Y)
    plt.show()    
    if 'path' in optional:
        f.savefig(optional['path']+'.jpg', format='jpg', dpi=1200)
    plt.close()
    return()

#Cycles through all the regions within a spectro dictionary
def show_mregions(rectangles, spectros):
    for i,d in rectangles.items():
        AD1.show_region(rectangles, spectros, i)
        input('Press enter to continue')
    return()
    
#Makes a wavelet plot
def wave_plot(data, wavelet):
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(data, wavelet)
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

#creates a new template
def create_template(file_name, timestep, region_num, bat_name, **optional): #creates three templates (image, rectangle and array)
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD1.set_path()
    if 'dml' in optional and optional['dml']: #present and true
        path+='/Templates_dml' #add this to the pathway
    elif 'eval' in optional and optional['eval']: #present and true
        path+='/Templates_eval' #add this to the pathway
    #create folders if need be
    AD1.make_folders(path)
    #extract regions and rectangles
    list_bats, _=AD1.set_batscolor()
    num_bats, _=AD1.set_numbats(list_bats, **optional)
    rectangles, regions, _=AD2.spect_loop(file_name, **optional)
    #make hash-codes
    hash_code=hash(str(regions[int(timestep)][region_num]))
    #save image
    path_image=path + '/Templates_images/' + bat_name + '/' + str(hash_code) + '.png'
    plt.imshow(regions[int(timestep)][region_num], origin='lower')
    plt.savefig(path_image)
    plt.close()
    #build correct pathways
    path_array=path + '/Templates_arrays/' + bat_name + '/' + str(hash_code) + '.npy'
    path_rect=path + '/Templates_rect/' + bat_name + '/' + str(hash_code) + '.npy'
    #save the data
    np.save(path_array, regions[int(timestep)][region_num])
    np.save(path_rect, rectangles[int(timestep)][:, region_num])
    #print out the hash-codes to the user
    print('hash code:', hash_code)
    return()
    