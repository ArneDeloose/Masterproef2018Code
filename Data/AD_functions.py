from __future__ import division #changes / to 'true division'
import scipy.io.wavfile
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import AD_functions as AD
import pywt
import matplotlib.patches as patches
from skimage.measure import compare_ssim as ssim
from sklearn import manifold
import os
from scipy.cluster.hierarchy import dendrogram, linkage
#from sklearn.metrics.pairwise import euclidean_distances

def loading_init(**optional): #loads in certain things so they only run once
    regions_temp, rectangles_temp=AD.read_templates(**optional)
    list_bats, colors_bat=AD.set_batscolor(**optional)
    num_bats, num_total=AD.set_numbats(list_bats, **optional)
    freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats=AD.set_batfreq(rectangles_temp, regions_temp, list_bats, num_bats)
    return(freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp)

def adjustable_parameters():
    path=AD.set_path()
    f=open(path+'\parameters.txt', 'r')
    a=f.readlines()
    words=[None]*len(a)
    i=0
    for line in a:
        words[i]=line.split(';')
        i+=1
    binary_thresh=int(words[0][0])
    spec_min=int(words[1][0])
    spec_max=int(words[2][0])
    thresh=float(words[3][0])
    spect_window=int(words[4][0])
    spect_window_overlap=int(words[5][0])
    max_roi=int(words[6][0])
    w_1=float(words[7][0])
    w_2=float(words[8][0])
    w_3=float(words[9][0])
    w_4=float(words[10][0])
    w_impor=(w_1, w_2, w_3, w_4)
    network_dim1=int(words[11][0])
    network_dim2=int(words[12][0])
    context_window=int(words[13][0])
    context_window_freq=int(words[13][0])
    para=(binary_thresh, spec_min, spec_max, thresh, spect_window, spect_window_overlap, max_roi, w_impor, \
          network_dim1, network_dim2, context_window, context_window_freq)
    return(para)

def set_parameters():
    adj_para=AD.adjustable_parameters()
    X=int(adj_para[0]*255/100) #Threshold for noise binary image
    kern=3 #window for roi
    min_spec_freq=int(adj_para[1]/0.375) #freq to pixels
    max_spec_freq=int(adj_para[2]/0.375)
    network_dim = (adj_para[8], adj_para[9])
    n_iter = 10000
    init_learning_rate = 0.01
    normalise_data = False
    normalise_by_column = False
    context_window_freq=int(adj_para[11]/0.375)
    fig_size=(10,10)
    #1 point= 0.32 ms
    #1 point= 375 Hz
    para=(X, kern, adj_para[6], adj_para[3], min_spec_freq, max_spec_freq, adj_para[4], adj_para[5], adj_para[7], \
          network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column, adj_para[10], context_window_freq, fig_size)
    return(para)

def set_path():
    path=os.getcwd()
    return(path)
    
def spect(file_name, **optional):
    para=AD.set_parameters()
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

def spect_plot(samples, sample_rate, **optional):
    #Makes a spectrogram, data normalised to the range [0-1]
    #Change parameters of spectrogram (window, resolution)
    para=AD.set_parameters()
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
        spectro=AD.substraction(dummy[min_spec_freq:max_spec_freq,:])
    return(spectro) 

def substraction(spect):
    dummy=np.zeros([len(spect),len(spect[0,:])])
    spectro=np.maximum(dummy, spect-spect.mean()) #Take maximum of zero and value to remove negative values
    spectro=np.array(spectro, dtype=np.uint8) #Convert back from float to uint8
    return(spectro)

def spect_loop(file_name, **optional): 
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Empty images are skipped
    para=AD.set_parameters()
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
    sample_rate, samples, _, _,steps= AD.spect('Audio_data/'+ file_name, **optional);
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
        spectros[i]=AD.spect_plot(samples_dummy, sample_rate, **optional)
        ctrs, dummy_flag=AD.ROI(spectros[i], kern, X)
        if dummy_flag:
            rectangles[i], regions[i]=AD.ROI2(ctrs, spectros[i])
            rectangles, regions=AD.check_overlap(rectangles, regions, spectros, i, spect_window, spect_window_overlap)
    rectangles2, regions2=AD.overload(rectangles, regions, **optional)
    regions3=AD.rescale_region(regions2)
    return(rectangles2, regions3, spectros)

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

def rescale_region(reg):
    region={}
    for i,d in reg.items():
        region[i]={}
        for j,d in reg[i].items():
            dummy=(reg[i][j]-reg[i][j].min())/(reg[i][j].max()-reg[i][j].min()) 
            region[i][j]=np.array(np.round(dummy*256), dtype=np.uint8) #Convert to grayscale
    return(region)

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

def overload(rectangles, regions, **optional): #deletes entries with ten or more rectangles
    para=AD.set_parameters()
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

def show_region(rectangles, spectros, i, **optional):
    para=AD.adjustable_parameters()
    if 'spec_min' in optional:
        spec_min=optional['spec_min']
    else:
        spec_min=para[1]
    if 'spec_max' in optional:
        spec_max=optional['spec_max']
    else:
        spec_max=para[2]
    if 't_max' in optional:
        t_max=optional['t_max']
    else:
        t_max=para[4]
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

def show_mregions(rectangles, spectros):
    for i,d in rectangles.items():
        AD.show_region(rectangles, spectros, i)
        input('Press enter to continue')
    return()

def compare_img(img1, img2, rectangle, min_freq, max_freq):
    si=(len(img2[0,:]), len(img2))
    img1_new=cv2.resize(img1, si)
    score=ssim(img1_new, img2, multichannel=True)
    if (rectangle[1]+rectangle[3]/2)<min_freq or (rectangle[1]+rectangle[3]/2)>max_freq:
        score=-1 #set to minimum score
    return(score)

def compare_img2(img1, img2): #same as compare_img but doesn't use freq
    #And always scales according to largest image
    a=len(img1[0,:])
    b=len(img1)
    c=len(img2[0,:])
    d=len(img2)
    if a*b==c*d: #same size
        if a>c: #img2->img1
            si=(a,b)
            img2_new=cv2.resize(img2, si)
            score=ssim(img1, img2_new, multichannel=True)
        else: #img1->img2
            si=(c,d)
            img1_new=cv2.resize(img1, si)
            score=ssim(img1_new, img2, multichannel=True)
    else: #different size
        if a*b>c*d: #1st image is bigger
            si=(a,b)
            img2_new=cv2.resize(img2, si)
            score=ssim(img1, img2_new, multichannel=True)
        else:
            si=(c,d)
            img1_new=cv2.resize(img1, si)
            score=ssim(img1_new, img2, multichannel=True)
    return(score)

def resize_img_plot(img1,img2):
    si=(len(img2[0,:]), len(img2))
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

def calc_sim_matrix(rectangles, regions): #calculates sim_matrices
    sim_mat1=np.zeros((len(regions), len(regions))) #ssim
    sim_mat2=np.zeros((len(regions), len(regions))) #freq_range_sq
    sim_mat3=sim_mat2.copy() #freq_lowest_sq
    sim_mat4=sim_mat2.copy() #freq_highest_sq
    sim_mat5=sim_mat2.copy() #freq_av_sq    
    sim_mat6=sim_mat2.copy() #duration_sq
    for i,j in regions.items():
        for k,l in regions.items():
            sim_mat1[i,k]=AD.compare_img2(regions[i], regions[k])
            sim_mat2[i,k]=(rectangles[3,i]-rectangles[3,k])**2
            sim_mat3[i,k]=(rectangles[1,i]-rectangles[1,k])**2
            sim_mat4[i,k]=((rectangles[1,i]+rectangles[3,i])-(rectangles[1,k]+rectangles[3,k]))**2
            sim_mat5[i,k]=((rectangles[1,i]+rectangles[3,i]/2)-(rectangles[1,k]+rectangles[3,k]/2))**2
            sim_mat6[i,k]=(rectangles[2,i]-rectangles[2,k])**2
    sim_mat1=(1-sim_mat1)/2 #0-1
    return(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6)

def calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, m_weight):
    w1,w2,w3,w4,w5,w6=AD.set_weights(m_weight)
    dist_mat=(w1*sim_mat1)+(w2*sim_mat2)+(w3*sim_mat3)+(w4*sim_mat4)+(w4*sim_mat4)+(w4*sim_mat4)
    return(dist_mat)

def set_mweights(weight):
    if weight==0:
        w1=1
        w2=1/144
        w3=1/400
        w4=1/400
        w5=1/300
        w6=1/20
    elif weight==1: #drop first weight
        w1=0
        w2=1/144
        w3=1/400
        w4=1/400
        w5=1/300
        w6=1/20
    elif weight==2:
        w1=1
        w2=0
        w3=1/400
        w4=1/400
        w5=1/300
        w6=1/20
    elif weight==3:
        w1=1
        w2=1/144
        w3=0
        w4=1/400
        w5=1/300
        w6=1/20
    elif weight==4:
        w1=1
        w2=1/144
        w3=1/400
        w4=0
        w5=1/300
        w6=1/20
    elif weight==5:
        w1=1
        w2=1/144
        w3=1/400
        w4=1/400
        w5=0
        w6=1/20
    elif weight==6:
        w1=1
        w2=1/144
        w3=1/400
        w4=1/400
        w5=1/300
        w6=0
    return(w1,w2,w3,w4,w5,w6)
    
def calc_pos(dist_mat):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(dist_mat).embedding_
    return(pos)

def calc_pos_TSNE(dist_mat):
    seed = np.random.RandomState(seed=3)
    tsne = manifold.TSNE(n_components=2, n_iter=3000, min_grad_norm=1e-9, random_state=seed,
                   metric="precomputed")
    pos = tsne.fit(dist_mat).embedding_
    return(pos)

def plot_MDS(pos):
    s = 10
    plot1=plt.scatter(pos[0:38, 0], pos[0:38, 1], color='turquoise', s=s, lw=0, label='ppip')
    plot2=plt.scatter(pos[39:55, 0], pos[39:55, 1], color='red', s=s, lw=0, label='eser')
    plot3=plt.scatter(pos[56:61, 0], pos[56:61, 1], color='green', s=s, lw=0, label='mdau')
    plot4=plt.scatter(pos[62:79, 0], pos[62:79, 1], color='blue', s=s, lw=0, label='pnat')
    plot5=plt.scatter(pos[80:85, 0], pos[80:85, 1], color='orange', s=s, lw=0, label='nlei')
    #plot6=plt.scatter(pos[86:95, 0], pos[86:95, 1], color='black', s=s, lw=0, label='noise')
    plt.legend(handles=[plot1,plot2, plot3, plot4, plot5])
    plt.show()
    return()

def plot_MDS2(pos, dim1, dim2):
    s=10
    neur=int(dim1*dim2)
    plot1=plt.scatter(pos[0:neur, 0], pos[0:neur, 1], color='red', s=s, lw=0, label='neurons')
    plot2=plt.scatter(pos[neur+1:, 0], pos[neur+1:, 1], color='black', s=s, lw=0, label='data')
    plt.legend(handles=[plot1,plot2])
    plt.show()
    return()
    
def run_MDS(m_weight):
    rectangles_final, regions_final=AD.set_templates2()
    sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6=AD.calc_sim_matrix(rectangles_final, regions_final)
    dist_mat=AD.calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, m_weight)
    pos=AD.calc_pos(dist_mat)
    AD.plot_MDS(pos)
    return()

def run_TSNE(m_weight):
    rectangles_final, regions_final=AD.set_templates2()
    sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6=AD.calc_sim_matrix(rectangles_final, regions_final)
    dist_mat=AD.calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, m_weight)
    pos=AD.calc_pos_TSNE(dist_mat)
    AD.plot_MDS(pos)
    return()

def calc_features(rectangles, regions, templates, num_reg, list_bats, num_total):
    #regions and rectangles can have different index because regions can be deleted if they overlap with the previous spectrogram
    features=np.zeros((len(templates)+7, num_reg))
    features_freq=np.zeros((7, num_reg)) #unscaled freq info
    count=0
    k=0 #index of the rectangle
    features_key={}
    for i,d in regions.items():
        k=0 #reset index every timestep
        for j,d in regions[i].items():
            features_key[count]=(i,j)
            features[0, count]=rectangles[i][3,k] #freq range
            features[1, count]=rectangles[i][1,k] #min freq
            features[2, count]=rectangles[i][1,k]+rectangles[i][3,k] #max freq
            features[3, count]=rectangles[i][1,k]+rectangles[i][3,k]/2 #av freq
            features[4, count]=rectangles[i][2,k] #duration
            index=np.argmax(regions[i][j]) #position peak frequency
            l=len(regions[i][j][0,:]) #number of timesteps
            a=index%l #timestep at peak freq
            b=math.floor(index/l) #frequency at peak freq
            features[5, count]=a/l #peak frequency T
            features[6, count]=b+rectangles[i][1,k] #peak frequency F
            for l in range(len(templates)):
                features[l+7, count]=AD.compare_img2(regions[i][j], templates[l])
            features_freq[:, count]=features[:7, count]
            count+=1
            k+=1
    #Feature scaling, half of the clustering is based on freq and time information
    #for m in range(7):
        #features[m,:]=(num_total/7)*(features[m, :]-features[m, :].min())/(features[m, :].max()-features[m, :].min())
    return(features, features_key, features_freq)

#variant function for single dictionaries (non-nested), used in DML code
def calc_features2(rectangles, regions, templates, list_bats, num_total):
    #regions and rectangles can have different index because regions can be deleted if they overlap with the previous spectrogram
    num_reg=len(regions)
    features=np.zeros((len(templates)+7, num_reg))
    for i in range(len(regions)):
        features[0, i]=rectangles[i][3] #freq range
        features[1, i]=rectangles[i][1] #min freq
        features[2, i]=rectangles[i][1]+rectangles[i][3] #max freq
        features[3, i]=rectangles[i][1]+rectangles[i][3]/2 #av freq
        features[4, i]=rectangles[i][2] #duration
        index=np.argmax(regions[i]) #position peak frequency
        l=len(regions[i][0, :]) #number of timesteps
        a=index%l #timestep at peak freq
        b=math.floor(index/l) #frequency at peak freq
        features[5, i]=a/l #peak frequency T
        features[6, i]=b+rectangles[i][1] #peak frequency F
        for l in range(len(templates)):
            features[l+7, i]=AD.compare_img2(regions[i], templates[l])
    return(features)

def calc_num_regions(regions):
    num_reg=0
    for i,d in regions.items():
        for j,d in regions[i].items():
            num_reg+=1
    return(num_reg)

def calc_col_labels(features, features_freq, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, **optional): #based upon percentage scores
    label_colors={}
    per_total={}
    per_total2={}
    para=AD.set_parameters()
    if 'thresh' in optional:
        thresh=optional['thresh']
    else:
        thresh=para[3]
    if 'w_impor' in optional:
        w_impor=optional['w_impor']
    else:
        w_impor=para[8]
    for i in range(features.shape[1]): #check columns one by one
        count=np.zeros((len(list_bats),)) #counters per
        count2=np.zeros((len(list_bats),)) #counters reg
        per=np.zeros((len(list_bats),)) #percentage scores
        per2=np.zeros((len(list_bats),)) #percentage scores reg
        weight=0
        dummy=(features[5:, i]>thresh) #matching bats
        for j in range(len(dummy)):
            lower_bound=0
            upper_bound=num_bats[0]
            for k in range(len(list_bats)):
                if k==0: #k-1 doesn't exist at first
                    if j<=upper_bound and dummy[j]==True: #match
                        weight=AD.col_weight(features_freq, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, i, k, w_impor)
                        count[k]+=(1/weight)
                        count2[k]+=1
                else: #every other k
                    lower_bound+=num_bats[k-1]
                    upper_bound+=num_bats[k]
                    if lower_bound<j<=upper_bound and dummy[j]==True: #match
                        weight=AD.col_weight(features_freq, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, i, k, w_impor)
                        count[k]+=(1/weight)
                        count2[k]+=1
            if sum(count)>0: #there are matches
                per=count/num_bats
                per2=count2/num_bats
                dummy_index=np.argmax(per) #index max per_score
                label_colors[i]=colors_bat[list_bats[dummy_index]]
            else: #no matches
                label_colors[i]= "#000000" #black 
            per_total[i]=per
            per_total2[i]=per2
    return(label_colors, per_total, per_total2)

def col_weight(features_freq, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, i, k, w_impor):
    weight1=(features_freq[1, i]+features_freq[3, i]-freq_bats[k])**2
    weight2=(features_freq[0, i]-freq_range_bats[k])**2
    weight3=(features_freq[5, i]-freq_peakT_bats[k])**2
    weight4=(features_freq[6, i]-freq_peakF_bats[k])**2
    weight=1+(weight1*w_impor[0])+(weight2*w_impor[1])+(weight3*w_impor[2])+(weight4*w_impor[3])
    return(weight)

def set_numbats(list_bats, **optional): #sets the number of templates per bat
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD.set_path()
    AD.make_folders(path)
    full_path='' #will be overwritten every time
    num_bats=np.zeros((len(list_bats),), dtype=np.uint8)
    for i in range(len(list_bats)):
        path2=list_bats[i]
        full_path=path + '/Templates_arrays/' + path2
        files_list= os.listdir(full_path) #list all templates in folder
        num_bats[i] = len(files_list) #number of templates
    num_total=sum(num_bats)
    return(num_bats, num_total)

def set_batfreq(rectangles_temp, regions_temp, list_bats, num_bats): #sets the lowest frequency of each bat
    #Change this to use rectangles instead
    freq_bats=[None] *len(list_bats)
    freq_range_bats=[None] *len(list_bats)
    freq_peakT_bats=[None] *len(list_bats) #relative time
    freq_peakF_bats=[None] *len(list_bats) #frequency
    max_index=0
    for i in range(len(list_bats)):
        tot_freq=[None] *num_bats[i]
        tot_freq_range=[None] *num_bats[i]
        tot_freq_peakT=[None] *num_bats[i]
        tot_freq_peakF=[None] *num_bats[i]
        for j in range(num_bats[i]):
            tot_freq[j]=rectangles_temp[max_index+j][1]+rectangles_temp[max_index+j][3]
            tot_freq_range[j]=rectangles_temp[max_index+j][3]
            index=np.argmax(regions_temp[max_index+j])
            l=len(regions_temp[max_index+j][0,:]) #timesteps
            a=index%l #timestep
            b=math.floor(index/l) #frequency
            tot_freq_peakT[j]=a/l #relative time
            tot_freq_peakF[j]=b+rectangles_temp[max_index+j][1]
        max_index+=j #keep counting
        freq_bats[i]=np.median(tot_freq)
        freq_range_bats[i]=np.median(tot_freq_range)
        freq_peakT_bats[i]=np.median(tot_freq_peakT)
        freq_peakF_bats[i]=np.median(tot_freq_peakF)
    return(freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats)

def set_batscolor(**optional): #dictionary linking bats to colors
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD.set_path()
    path=AD.set_path()
    colors_bat={}
    list_bats=os.listdir(path + '/Templates_arrays')
    colors=("#ff0000", "#008000", "#0000ff", "#a52a2a", "#ee82ee", 
            "#f0f8ff", "#faebd7", "#f0ffff", "#006400", "#ffa500",
            "#ffff00", "#40e0d0", "#4b0082", "#ff00ff", "#ffd700")
    for i in range(len(list_bats)):
        colors_bat[list_bats[i]]=colors[i]
    return(list_bats, colors_bat)
            
def plot_dendrogram(features, label_colors, **optional):
    features_new=np.transpose(features) #map only works on rows, not columns
    linked = linkage(features_new, 'average')
    plt.figure(figsize=(20, 14))
    dendrogram(linked)
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels() #gets labels
    for lbl in xlbls:
        lbl.set_color(label_colors[int(lbl.get_text())])#sets color label
    if 'name' in optional:
        plt.savefig('dendrograms/' + optional['name'] + '.png')
    else:
        plt.show()
    plt.close()
    return()

def show_region2(rectangles, spectros, features_key, i, **optional): #uses feature label
    (a,b)=features_key[i]
    para=AD.adjustable_parameters()
    if 'spec_min' in optional:
        spec_min=optional['spec_min']
    else:
        spec_min=para[1]
    if 'spec_max' in optional:
        spec_max=optional['spec_max']
    else:
        spec_max=para[2]
    if 't_max' in optional:
        t_max=optional['t_max']
    else:
        t_max=para[4]
    f, ax1 = plt.subplots()
    ax1.imshow(spectros[a], origin='lower')
    rect = patches.Rectangle((rectangles[a][0,b],rectangles[a][1,b]),
                                rectangles[a][2,b],rectangles[a][3,b],
                                linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    labels_X = [item.get_text() for item in ax1.get_xticklabels()]
    labels_Y = [item.get_text() for item in ax1.get_yticklabels()]
    labels_X[1]=0
    labels_X[2]=t_max
    for i in range(1, len(labels_Y)-1):
       labels_Y[i]=int((spec_max-spec_min)*(i-1)/(len(labels_Y)-3)+spec_min)
    ax1.set_xticklabels(labels_X)
    ax1.set_yticklabels(labels_Y)
    if 'name' in optional:
        plt.savefig(optional['name'] + '.png')
    else:
        plt.show()
    plt.close()
    return()

def hier_clustering(file_name, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp, **optional):  
    rectangles, regions, spectros=AD.spect_loop(file_name, **optional)
    num_reg=AD.calc_num_regions(regions)
    features, features_key, features_freq=AD.calc_features(rectangles, regions, templates, num_reg, list_bats, num_total)
    col_labels, per_total, per_total2=AD.calc_col_labels(features, features_freq, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats)
    if 'write' in optional:
        if optional['write']: #true
            AD.plot_dendrogram(features, col_labels, name=file_name)
        else: #false
            AD.plot_dendrogram(features, col_labels)
    else:
        AD.plot_dendrogram(features, col_labels)
    return(col_labels, features_key, rectangles, spectros, per_total, per_total2)

def write_output(list_files, **optional): #Optional only works on non TE data
    para=AD.set_parameters()
    if 'spect_window' in optional:
        spect_window=optional['spect_window']
    else:
        spect_window=para[6]
    if 'spect_window_overlap' in optional:
        spect_window_overlap=optional['spect_window_overlap']
    else:
        spect_window_overlap=para[7]
    #loading
    freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp=AD.loading_init(**optional)
    if 'full' in optional:
       if optional['full']: #True
           if 'Audio_data' in optional:
               path=optional['Audio_data']
               list_files2=os.listdir(path)
           else:
               path=AD.set_path()
               list_files2=os.listdir(path + '/Audio_data') 
           count=0
           for i in range(len(list_files2)):
               if list_files2[i-count][-4:]!='.WAV':
                   del list_files2[i-count] #delete files that aren't audio
                   count+=1 #len changes, take this into account
       else:
           list_files2=list_files
    else:
        list_files2=list_files
    list_bats, colors_bat=AD.set_batscolor()
    #Check directories
    if not os.path.exists('dendrograms'):
        os.makedirs('dendrograms')
    for k in range(len(list_bats)):
        if not os.path.exists(list_bats[k]):
            os.makedirs(list_bats[k])
    #write out color key
    dummy_index=range(len(list_bats))
    for i in range(len(list_bats)):
        plt.scatter(0, dummy_index[i], color=colors_bat[list_bats[i]])
        plt.annotate(list_bats[i], (0.001, dummy_index[i]))
    plt.savefig('dendrograms/color_key.png')
    plt.close()
    #create empty dictionaries
    col_labels={}
    features_key={}
    rectangles={}
    spectros={}
    per_total={}
    per_total2={}
    optional['write']=True
    #run clustering and save output    
    for i in range(len(list_files2)):
        col_labels[i], features_key[i], rectangles[i], spectros[i], per_total[i], per_total2[i]=AD.hier_clustering(str(list_files2[i]), freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp, **optional)
    total_count=np.zeros((len(list_bats), 1), dtype=np.uint8)
    #output file
    if 'results1' in optional and 'results2' in optional:
        if os.path.exists(optional['results1'] + '.txt'): #file already exists
            open(optional['results1']+ '.txt', 'w').close() #clear file
        f1=open(optional['results1']+ '.txt', 'a') #edit file
        if os.path.exists(optional['results2'] + '.txt'):
            open(optional['results2']+ '.txt', 'w').close() #clear file
        f2=open(optional['results2']+ '.txt', 'a') #edit file
    else: #no output file is given, create standard one
        if os.path.exists('results1.txt'):
            open('results1.txt', 'w').close() #clear file
        f1=open('results1.txt', 'a') #edit file
        if os.path.exists('results1.txt'):
            open('results2.txt', 'w').close() #clear file
        f2=open('results2.txt', 'a') #edit file
    for k in range(len(list_bats)):
        f1.write(str(list_bats[k]) +': ' + '\n'); #name bat
        f1.write('\n') #skip line
        for i in range(len(list_files2)):
            f1.write(str(i) + ': ' + str(list_files2[i]) + "\n") #name file
            count=0
            for j in range(len(col_labels[i])):
                if col_labels[i][j]==colors_bat[list_bats[k]]:
                    #print('k:', k)
                    #print('i:', i)
                    #print('j:', j)
                    f1.write('Timestep: ' + str(features_key[i][j][0]) + ', region: ' + str(features_key[i][j][1])
                    + ', score1: ' + str(int(100000*per_total[i][j][k])) + ' mil, score2: ' + str(int(100*per_total2[i][j][k])) + ' %'
                    + ', coordinates (x1, x2, y1, y2): ' + str(int(features_key[i][j][0]*(spect_window-spect_window_overlap)+rectangles[i][features_key[i][j][0]][0, features_key[i][j][1]]*0.32))
                    + '-' + str(int(features_key[i][j][0]*(spect_window-spect_window_overlap)+(rectangles[i][features_key[i][j][0]][0, features_key[i][j][1]]+rectangles[i][features_key[i][j][0]][2,features_key[i][j][1]])*0.32)) + ' ms, '
                    + str(int(rectangles[i][features_key[i][j][0]][1, features_key[i][j][1]]*0.375)) + '-' 
                    + str(int((rectangles[i][features_key[i][j][0]][1, features_key[i][j][1]]+rectangles[i][features_key[i][j][0]][3, features_key[i][j][1]])*0.375)) + ' kHz' + ' \n');                    
                    count+=1
                    temp_str=list_bats[k] + '/timestep_' + str(features_key[i][j][0]) + '_region_' + str(features_key[i][j][1]) + '_file_' + str(list_files2[i])
                    show_region2(rectangles[i], spectros[i], features_key[i], j, name=temp_str)
            f1.write('\n') #empty line between different files
            total_count[k]+=count
    for k in range(len(list_bats)):
        f2.write(str(list_bats[k]) +': ' + str(total_count[k]) + '\n');
    f1.close()
    f2.close()
    return()

def calc_output(list_files, net, **optional): #Optional only works on non TE data
    #loading
    freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp=AD.loading_init(**optional)
    if 'full' in optional:
       if optional['full']: #True
           if 'Audio_data' in optional:
               path=optional['Audio_data']
               list_files2=os.listdir(path)
           else:
               path=AD.set_path()
               list_files2=os.listdir(path + '/Audio_data') 
           count=0
           for i in range(len(list_files2)):
               if list_files2[i-count][-4:]!='.WAV':
                   del list_files2[i-count] #delete files that aren't audio
                   count+=1 #len changes, take this into account
       else:
           list_files2=list_files
    else:
        list_files2=list_files
    list_bats, colors_bat=AD.set_batscolor()
    #Check directories
    if not os.path.exists('dendrograms'):
        os.makedirs('dendrograms')
    for k in range(len(list_bats)):
        if not os.path.exists(list_bats[k]):
            os.makedirs(list_bats[k])
    #create empty dictionaries
    net_label={}
    features={}
    features_key={}
    features_freq={}
    rectangles={}
    regions={}
    spectros={}
    optional['write']=True
    #run clustering and save output    
    for i in range(len(list_files2)):
        rectangles[i], regions[i], spectros[i]=AD.spect_loop(list_files2[i], **optional)
        num_reg=AD.calc_num_regions(regions[i])
        features[i], features_key[i], features_freq[i]=AD.calc_features(rectangles[i], regions[i], templates, num_reg, list_bats, num_total)
        net_label[i]=AD.calc_BMU_scores(features[i], net, **optional)
    return(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2)   

def rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net, **optional):
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD.set_parameters()
        network_dim= para[9]
    #initialisation
    temp_rectangle={}
    temp_region={}
    temp_spectro={}
    temp_name={}
    temp_key=()
    full_rectangle={}
    full_region={}
    full_spectro={}
    full_name={}
    temp_key2=0
    for i in range(network_dim[0]):
        full_rectangle[i]={}
        full_region[i]={}
        full_spectro[i]={}
        full_name[i]={}
        for j in range(network_dim[1]):
            full_rectangle[i][j]={}
            full_region[i][j]={}
            full_spectro[i][j]={}
            full_name[i][j]={}
            count=0
            dist_temp=[]
            for k in range(len(net_label)): #file
                for l in range(len(net_label[k])): #number
                    if net_label[k][l][0]==i and net_label[k][l][1]==j:
                        temp_key=features_key[k][l]
                        temp_region[count]=regions[k][temp_key[0]][temp_key[1]]
                        temp_key2=AD.check_key(regions[k][temp_key[0]], temp_key[1])
                        temp_rectangle[count]=rectangles[k][temp_key[0]][:, temp_key2]
                        temp_spectro[count], extra_time=AD.calc_context_spec(spectros, k, temp_key)
                        temp_rectangle[count][0]+=extra_time #correct for bigger spect
                        #temp_spectro[count]=spectros[k][temp_key[0]]
                        distance=sum((features[k][:,l] - net[i,j,:])**2)
                        temp_name[count]=list_files2[k] + ', timestep: ' + str(temp_key[0]) + ', region: ' + str(temp_key[1]) + ', distance: ' + str(distance) 
                        #calc distance and save this
                        dist_temp.append(distance)
                        count+=1
            temp_order=np.argsort(dist_temp) #index according to distance
            for m in range(len(temp_order)):
                full_region[i][j][m]=temp_region[temp_order[m]]
                full_rectangle[i][j][m]=temp_rectangle[temp_order[m]]
                full_spectro[i][j][m]=temp_spectro[temp_order[m]]
                full_name[i][j][m]=temp_name[temp_order[m]]
    return(full_region, full_rectangle, full_spectro, full_name)

def check_key(regions, temp_key): #matches indexes rectangles and regions
    count_dummy=0
    for i in range(temp_key+1):
        if str(i) in regions.keys():
            count_dummy+=1
    return(count_dummy-1)

def calc_matching(full_name, **optional):
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD.set_parameters()
        network_dim= para[9]
    M=np.zeros((network_dim[0], network_dim[1]), dtype=np.uint8)
    for i in range(network_dim[0]):
        for j in range(network_dim[1]):
            M[i,j]=len(full_name[i][j])
    return(M)
    
def plot_region_neuron(full_region, full_rectangle, full_spectro, full_name, dim1, dim2, point, **optional):
    if 'context_window_freq' in optional:
        context_window_freq=optional['context_window_freq']
    else:
        para=AD.set_parameters()
        context_window_freq=para[15]
    if 'fig_size' in optional:
        fig_size=optional['fig_size']
    else:
        para=AD.set_parameters()
        fig_size=para[16]
    #set frequency cutoff
    freq1_index=full_rectangle[dim1][dim2][point][1]-context_window_freq
    if freq1_index<0:
        freq1_index=0
    freq2_index=full_rectangle[dim1][dim2][point][1]+full_rectangle[dim1][dim2][point][3]+context_window_freq
    if freq2_index>full_spectro[dim1][dim2][point].shape[0]:
        freq2_index=full_spectro[dim1][dim2][point].shape[0]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size,  gridspec_kw = {'width_ratios':[4, 1, 1]})
    
    #image 1 (full spectro)
    
    ax1.imshow(full_spectro[dim1][dim2][point][freq1_index:freq2_index], origin='lower', aspect='auto')
    rect = patches.Rectangle((full_rectangle[dim1][dim2][point][0], full_rectangle[dim1][dim2][point][1]-freq1_index),
                              full_rectangle[dim1][dim2][point][2], full_rectangle[dim1][dim2][point][3],
                              linewidth=1, edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    plt.draw() #sets up the ticks
    labels_Y = [item.get_text() for item in ax1.get_yticklabels()] #original labels
    labels_y=list() #new labels
    labels_y.append(labels_Y[0])
    for i in range(1, len(labels_Y)):
        labels_y.append(str(float((float(labels_Y[i])+freq1_index)*0.375)))
    labels_X = [item.get_text() for item in ax1.get_xticklabels()]
    labels_x=list()
    labels_x.append(labels_X[0])
    for i in range(1, len(labels_X)):
        labels_x.append(str(int(int(labels_X[i])*2.34375))) #convert to ms
    ax1.set_xticklabels(labels_x)
    ax1.set_yticklabels(labels_y)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (kHz)')
    
    #image 2 (detail spectro)
    
    ax2.imshow(full_region[dim1][dim2][point], origin='lower')
    
    #Image 3 (FI plot)
    
    FI_matrix=AD.calc_FI_matrix(full_region[dim1][dim2][point])
    ax3.imshow(FI_matrix, origin='lower', aspect='auto')
    plt.draw()
    
    labels_X = [item.get_text() for item in ax3.get_xticklabels()] #original labels
    labels_x=list() #new labels
    labels_x.append(labels_X[0])
    for i in range(1, len(labels_X)):
        labels_x.append(str(round((float(labels_X[i])+freq1_index+context_window_freq)*0.375, 2)))

    ax3.set_xticklabels(labels_y)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Intensity')

    #title
    ax1.set_title(full_name[dim1][dim2][point])
    
    plt.show()
    plt.close()
    return()

def calc_context_spec(spectros, k, temp_key, **optional): #add windows to spectrogram
    para=AD.set_parameters()
    if 'spect_window' in optional:
        spect_window=optional['spect_window']
    else:
        spect_window=para[6]
    if 'spect_window_overlap' in optional:
        spect_window_overlap=optional['spect_window_overlap']
    else:
        spect_window_overlap=para[7]
    if 'context_window' in optional:
        context_window=optional['context_window']
    else:
        context_window=para[14]
    overlap_factor=(1-spect_window_overlap/spect_window) #corrects for overlap between windows
    max_key=len(spectros[k])-1
    extra_time=0 #extra time points to correct rectangle 
    context_spec=spectros[k][temp_key[0]]
    steps=context_spec.shape[1]
    #left
    for i in range(1, context_window+1):
        if (temp_key[0]-i)>=0: #window the left exists
            context_spec=np.concatenate((spectros[k][temp_key[0]-i][:, 0:int(steps*overlap_factor)], context_spec), axis=1)
            extra_time+=int(len(spectros[k][temp_key[0]-i][0,:])*overlap_factor)
    #right
    for i in range(1, context_window+1):
        if (temp_key[0]+i)<=max_key: #window to the right exists
            context_spec=np.concatenate((context_spec, spectros[k][temp_key[0]+i][:, int(steps*(1-overlap_factor)):]), axis=1)
    return(context_spec, extra_time)

def calc_maxc(full_names, **optional):
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD.set_parameters()
        network_dim=para[9]
    max_c=0
    for i in range(network_dim[0]):
        for j in range(network_dim[1]):
            temp_c=len(full_names[i][j])
            if temp_c>max_c:
                max_c=temp_c
    return(max_c)

def create_template(file_name, timestep, region_num, bat_name, **optional): #creates three templates (image, rectangle and array)
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD.set_path()
    AD.make_folders(path)
    list_bats, _=AD.set_batscolor()
    num_bats, _=AD.set_numbats(list_bats, **optional)
    if not bat_name in list_bats: #bat already exists
        os.makedirs(path + '/Templates_arrays/' + bat_name)
        os.makedirs(path + '/Templates_images/' + bat_name)
        os.makedirs(path + '/Templates_rect/' + bat_name)
    rectangles, regions, _=AD.spect_loop(file_name, **optional)
    hash_image=hash(str(regions[int(timestep)][region_num]))
    hash_rect=hash(str(rectangles[int(timestep)][:, region_num]))
    path_image=path + '/Templates_images/' + bat_name + '/' + str(hash_image) + '.png'
    plt.imshow(regions[int(timestep)][region_num], origin='lower')
    plt.savefig(path_image)
    plt.close()
    path_array=path + '/Templates_arrays/' + bat_name + '/' + str(hash_image) + '.npy'
    path_rect=path + '/Templates_rect/' + bat_name + '/' + str(hash_rect) + '.npy'
    np.save(path_array, regions[int(timestep)][region_num])
    np.save(path_rect, rectangles[int(timestep)][:, region_num])
    print('hash code image:', hash_image)
    print('hash code array:', hash_rect)
    return()

def read_templates(**optional): #reads in templates from the path to the general folder
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD.set_path()
    full_path='' #string will be constructed every step
    full_path_rec=''
    list_bats, _=AD.set_batscolor()
    if 'Templates' in optional:
        num_bats, _=AD.set_numbats(list_bats, Templates=optional['Templates'])
    else:
        num_bats, _=AD.set_numbats(list_bats)
    regions={}
    rectangles={}
    count=0
    for i in range(len(list_bats)):
        list_files_arrays=os.listdir(path + '/Templates_arrays/' + list_bats[i]) #list all files in folder
        list_files_rect=os.listdir(path + '/Templates_rect/' + list_bats[i])
        count1=0
        count2=0
        for k in range(len(list_files_arrays)):
            if list_files_arrays[k-count1][-4:]!='.npy':
                del list_files_arrays[k-count1] #remove files that aren't npy extensions
                count1+=1 #len of list changes, take this into account
        for k in range(len(list_files_rect)): #repeat for rectangles
            if list_files_rect[k-count2][-4:]!='.npy':
                del list_files_rect[k-count2] #remove files that aren't npy extensions
                count2+=1 #len of list changes, take this into account
        for j in range(num_bats[i]): #go through the files one by one
            full_path=path+ '/Templates_arrays/' + list_bats[i] + '/' + list_files_arrays[j]
            full_path_rec=path+ '/Templates_rect/' + list_bats[i] + '/' + list_files_rect[j]
            regions[count]=np.load(full_path)
            rectangles[count]=np.load(full_path_rec)
            count+=1
    return(regions, rectangles)

def make_folders(path): #makes folders if they don't exist yet
    os.chdir(path)
    if not os.path.exists('Templates_arrays'):
        os.makedirs('Templates_arrays')
    if not os.path.exists('Templates_images'):
        os.makedirs('Templates_images')
    if not os.path.exists('Templates_rect'):
        os.makedirs('Templates_rect')
    return()

def import_map(map_name, **optional):
    if 'path' in optional:
        path=optional['path']
    else:
        path=AD.set_path()
    net=np.load(path+ '/' + map_name+ '.npy')
    raw_data=np.load(path+ '/' + map_name+ '_data.npy')
    return(net, raw_data)

def fit_SOM(list_files, **optional):
    if 'full' in optional: #read all files in folder
       if optional['full']: #True
           if 'Audio_data' in optional:
               path=optional['Audio_data']
               list_files2=os.listdir(path)
           else:
               path=AD.set_path()
               list_files2=os.listdir(path + '/Audio_data') 
           count=0
           for i in range(len(list_files2)):
               if list_files2[i-count][-4:]!='.WAV':
                   del list_files2[i-count] #delete files that aren't audio
                   count+=1 #len changes, take this into account
       else:
           list_files2=list_files
    else:
        list_files2=list_files
    #parameters
    para=AD.set_parameters()
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        network_dim= para[9]
    if 'n_iter' in optional:
        n_iter=optional['n_iter']
    else:
        n_iter= para[10]
    if 'init_learning_rate' in optional:
        init_learning_rate=optional['init_learning_rate']
    else:
        init_learning_rate= para[11]
    if 'normalise_data' in optional:
        normalise_data=optional['normalise_data']
    else:
        normalise_data= para[12]
    if 'normalise_by_column' in optional:
        normalise_by_column=optional['normalise_by_column']
    else:
        normalise_by_column= para[13]
    
    #data already present, override list_files
    if 'features' in optional:
        raw_data=optional['features']
    else:
        #first file
        rectangles1, regions1, spectros1=AD.spect_loop(list_files2[0])
        num_reg=AD.calc_num_regions(regions1)
        freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD.loading_init(**optional)
        features1, _, _=AD.calc_features(rectangles1, regions1, regions_temp, num_reg, list_bats, num_total)
        raw_data=np.zeros((features1.shape[0], 0))
        raw_data=np.concatenate((raw_data, features1), axis=1)
        #other files
        for i in range(1, len(list_files2)):
            rectangles1, regions1, spectros1=AD.spect_loop(list_files2[i])
            num_reg=AD.calc_num_regions(regions1)
            features1, features_key1, features_freq1=AD.calc_features(rectangles1, regions1, regions_temp, num_reg, list_bats, num_total)
            raw_data=np.concatenate((raw_data, features1), axis=1)
    net=AD.SOM(raw_data, network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column, **optional)
    return(net, raw_data)

def SOM(raw_data, network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column, **optional):
    if 'DML' in optional:
        D=optional['DML']
    else:
        D=np.identity(raw_data.shape[0])
    m = raw_data.shape[0]
    n = raw_data.shape[1]
    # initial neighbourhood radius
    init_radius = max(network_dim[0], network_dim[1]) / 2
    # radius decay parameter
    time_constant = n_iter / np.log(init_radius)
    if normalise_data:
        if normalise_by_column:
            # normalise along each column
            col_maxes = raw_data.max(axis=0)
            data = raw_data / col_maxes[np.newaxis, :]
        else:
            # normalise entire dataset
            data = raw_data / data.max()
    else:
        data=raw_data
    # setup random weights between 0 and 1
    # weight matrix needs to be one m-dimensional vector for each neuron in the SOM
    net = np.random.random((network_dim[0], network_dim[1], m))
    for i in range(n_iter):
        # select a training example at random
        t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))    
        # find its Best Matching Unit
        bmu, bmu_idx = AD.find_bmu(t, net, m, D)
        # decay the SOM parameters
        r = AD.decay_radius(init_radius, i, time_constant)
        l = AD.decay_learning_rate(init_learning_rate, i, n_iter)
        # now we know the BMU, update its weight vector to move closer to input
        # and move its neighbours in 2-D space closer
        # by a factor proportional to their 2-D distance from the BMU
        for x in range(net.shape[0]):
            for y in range(net.shape[1]):
                w = net[x, y, :].reshape(m, 1)
                # get the 2-D distance (again, not the actual Euclidean distance)
                w_dist = np.sum((np.array([x, y]) - bmu_idx)**2)
                # if the distance is within the current neighbourhood radius
                if w_dist <= r**2:
                    # calculate the degree of influence (based on the 2-D distance)
                    influence = AD.calculate_influence(w_dist, r)
                    # now update the neuron's weight using the formula:
                    # new w = old w + (learning rate * influence * delta)
                    # where delta = input vector (t) - old w
                    new_w = w + (l * influence * (t - w))
                    # commit the new weight
                    net[x, y, :] = new_w.reshape(1, m)
    if 'export' in optional:
        path=AD.set_path()
        np.save(path + '/' + optional['export'] + '.npy', net)
        np.save(path + '/' + optional['export'] + '_data.npy', raw_data)
    return(net)

def find_bmu(t, net, m, D):
    #Find the best matching unit for a given vector, t, in the SOM
    #Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
     #            and bmu_idx is the index of this vector in the SOM
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max    
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum(np.dot(D, (w - t)**2))
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

def decay_radius(initial_radius, i, time_constant):
    return (initial_radius * np.exp(-i / time_constant))

def decay_learning_rate(initial_learning_rate, i, n_iter):
    return (initial_learning_rate * np.exp(-i / n_iter))

def calculate_influence(distance, radius):
    return (np.exp(-distance / (2* (radius**2))))

def calc_Umat(net):
    m = net.shape[0]
    n = net.shape[1]
    U=np.zeros((m, n), dtype=np.float)
    for i in range(m):
        for j in range(n):
            if i==0:
                if j==0:
                    U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j+1,:])+np.linalg.norm(net[i,j,:]-net[i+1,j,:])
                elif j==n-1:
                    U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j-1,:])+np.linalg.norm(net[i,j,:]-net[i+1,j,:])
                else:
                    U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j+1,:])+np.linalg.norm(net[i,j,:]-net[i+1,j,:])+ \
                    np.linalg.norm(net[i,j,:]-net[i,j-1,:])
            elif i==m-1:
                if j==0:
                    U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j+1,:])+np.linalg.norm(net[i,j,:]-net[i-1,j,:])
                elif j==n-1:
                    U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j-1,:])+np.linalg.norm(net[i,j,:]-net[i-1,j,:])
                else:
                    U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j+1,:])+np.linalg.norm(net[i,j,:]-net[i-1,j,:])+ \
                    np.linalg.norm(net[i,j,:]-net[i,j-1,:])
            elif j==0:
                U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j+1,:])+np.linalg.norm(net[i,j,:]-net[i+1,j,:])+ \
                    np.linalg.norm(net[i,j,:]-net[i-1,j,:])
            elif j==n-1:
                U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j-1,:])+np.linalg.norm(net[i,j,:]-net[i-1,j,:])+ \
                    np.linalg.norm(net[i,j,:]-net[i+1,j,:])
            else:
                U[i,j]=np.linalg.norm(net[i,j,:]-net[i,j+1,:])+np.linalg.norm(net[i,j,:]-net[i,j-1,:])+ \
                np.linalg.norm(net[i,j,:]-net[i+1,j,:])+np.linalg.norm(net[i,j,:]-net[i-1,j,:])
    return(U)

def calc_BMU_scores(data, net, **optional):
    m=data.shape[0] #number of features
    n=data.shape[1] #number of data points
    score_BMU=np.zeros((n, 2), dtype=np.uint8)
    if 'DML' in optional:
        D=optional['DML']
    else:
        D=np.identity(m)
    for i in range(n):
        t = data[:, i].reshape(np.array([m, 1]))    
        _, bmu_idx=AD.find_bmu(t, net, m, D)
        score_BMU[i, 0]=bmu_idx[0]
        score_BMU[i, 1]=bmu_idx[1]
    return(score_BMU)

def calc_net_features(net, **optional): #transforms network features to more suitable form
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD.set_parameters()
        network_dim= para[9]
    net_features=np.zeros((net.shape[2], network_dim[0]*network_dim[1]))
    count=0
    for i in range(network_dim[0]):
        for j in range(network_dim[1]):
            net_features[:, count]=net[i, j, :]
            count+=1
    return(net_features)

def calc_dist_matrix2(net_features, axis, **optional): #calculates distance per column (if axis=1)
    if 'raw_data' in optional:
        array=np.concatenate((net_features, optional['raw_data']), axis=1)
    else:
        array=net_features
    D=np.zeros((array.shape[axis], array.shape[axis]), dtype=np.float)
    for i in range(array.shape[axis]):
        for j in range(array.shape[axis]):
            D[i,j]=sum((array[:, i]-array[:,j])**2)
    return(D)

def cor_plot(features, index, **optional): #index: start and stop index to make the plot
    fig = plt.figure(figsize=(8, 6))
    features_red=np.transpose(features[0][int(index[0]):int(index[1]), :])
    for i in range(1, len(features)):
        dummy=np.transpose(features[i][int(index[0]):int(index[1]), :])
        features_red=np.concatenate((features_red, dummy), axis=0)
    data = pd.DataFrame(data=features_red)
    correlations = data.corr()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)    
    plt.show()
    if 'export' in optional:
        fig.savefig(optional['export']+'.jpg', format='jpg', dpi=1200)
    plt.close
    return(correlations)

def plot_U(net, **optional):
    U=AD.calc_Umat(net)
    f, ax1 = plt.subplots()
    plt.imshow(U)
    plt.colorbar()
    plt.show()
    if 'export' in optional:
        f.savefig(optional['export']+ '.jpg', format='jpg', dpi=1200)
    plt.close()
    return()

def heatmap_neurons(M, **optional):
    f, ax1 = plt.subplots()
    plt.imshow(M)
    plt.colorbar()
    plt.show()
    if 'export' in optional:
        f.savefig(optional['export']+ '.jpg', format='jpg', dpi=1200)
    plt.close()
    return()

def calc_FI_matrix(spectro):
    FI_matrix=np.zeros((256, spectro.shape[0]), dtype=np.uint8)
    for i in range(spectro.shape[0]):
        for j in range(spectro.shape[1]):
            dummy=spectro[i,j] #intensity
            FI_matrix[dummy, i]+=1
    return(FI_matrix[1:, :])

def print_features(**optional):
    list_bats, colors_bat=AD.set_batscolor(**optional)
    num_bats, num_total=AD.set_numbats(list_bats, **optional)
    a=6
    print('Frequency: 0-'+str(a))
    for i in range(len(list_bats)):
        a+=1
        print(list_bats[i] + ': ' + str(a) + '-' + str(a+num_bats[i]))
        a+=num_bats[i]
    return()
    