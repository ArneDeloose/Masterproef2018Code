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
from sklearn import manifold
import os
from scipy.cluster.hierarchy import dendrogram, linkage
#from sklearn.metrics.pairwise import euclidean_distances

def adjustable_parameters():
    binary_thresh=10 #Threshold for conversion to binary image. Given as a percentage (default: 10%)
    spec_min=20 #minimum frequency in a spectrogram, given in kHz (default: 20 kHz)
    spec_max=120 #maximum frequency in a spectrogram, given in kHz (default: 120 kHz)
    thresh=0.65 #threshold for SSIM classification, number between -1 and 1 (default: 0.65)
    spect_window=200 #length of the local scaling window in ms (default: 200 ms)
    spect_overlap_window=50 #overlap between windows in ms (default: 50 ms)
    max_roi=10 #Maximum number of regions in a single spectrogram. (default: 10) Remember that the size of a spectrogram changes with spect_window
    para=(binary_thresh, spec_min, spec_max, thresh, spect_window, spect_overlap_window, max_roi)
    return(para)

def set_parameters():
    adj_para=AD.adjustable_parameters()
    X=int(adj_para[0]*255) #Threshold for noise binary image
    kern=[3,3] #window for roi
    min_spec_freq=int(adj_para[1]/0.375)
    max_spec_freq=int(adj_para[2]/0.58)
    #1 point= 0.58 ms
    #1 point= 375 Hz
    para=(X, kern, adj_para[6], adj_para[3], min_spec_freq, max_spec_freq, adj_para[4], adj_para[5])
    return(para)

def set_path():
    path=os.getcwd()
    return(path)
    
def set_freqthresh(num_class): #frequency number depends on minimum frequency used for the spectrum
    para=AD.set_parameters()
    min_spec_freq=para[4]
    max_spec_freq=para[5]    
    if num_class==0: #ppip
        min_freq=115-min_spec_freq #43 kHz
        max_freq=135-min_spec_freq #50 kHz
    elif num_class==1: #eser    
        min_freq=73-min_spec_freq #27.5 kHz
        max_freq=93-min_spec_freq #35 kHz
    else: #full spectro
        min_freq=0
        max_freq=max_spec_freq-min_spec_freq #max freq
    return(min_freq, max_freq)
        
def spect(file_name, **optional):
    para=AD.set_parameters()
    spect_window=para[6]
    spect_window_overlap=para[7]
    #Reads information from audio file
    [sample_rate,samples]=scipy.io.wavfile.read(file_name, mmap=False);
    if 'channel' in optional:
        if optional['channel']=='l':
            samples=samples[:,0]
        elif optional['channel']=='r':
            samples=samples[:,1]
    N=len(samples); #number of samples
    t=np.linspace(0,N/sample_rate, num=N); #time_array
    total_time=N/sample_rate;
    steps=math.floor(total_time/(spect_window+spect_window_overlap))
    return(sample_rate, samples, t, total_time, steps)

def spect_plot(samples, sample_rate):
    #Makes a spectrogram, data normalised to the range [0-1]
    #Change parameters of spectrogram (window, resolution)
    para=AD.set_parameters()
    min_spec_freq=para[4]
    max_spec_freq=para[5]
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=('hamming'), nfft=1024)
    dummy=(spectrogram-spectrogram.min())/(spectrogram.max()-spectrogram.min())
    dummy=np.array(np.round(dummy*256), dtype=np.uint8) #Convert to grayscale
    spectro=AD.substraction(dummy[min_spec_freq:max_spec_freq,:])
    return(spectro) 

def substraction(spect):
    dummy=np.zeros([len(spect),len(spect[0,:])])
    spectro=np.maximum(dummy, spect-spect.mean()) #Take maximum of zero and value to remove negative values
    spectro=np.array(spectro, dtype=np.uint8) #Convert back from float to uint8
    return(spectro)

def spect_loop(file_name, **optional): #hybrid code, one plot for 100 ms
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    para=AD.set_parameters()
    X=para[0]
    kern=para[1]
    spect_window=para[6]
    spect_overlap_window=para[7]
    #change number of steps so it matches the window and overlap
    #change algorithm so the spectrograms aren't fixed at 100 ms, but rather at the number of actual steps
    #templates are already defined so the algorthitm becomes obsolete
    rectangles={};
    regions={};
    spectros={};
    if 'channel' in optional: #time dilation
        sample_rate, samples, _, _,steps= AD.spect('Audio_data/'+ file_name, channel=optional['channel']);
        sample_rate=10*sample_rate
        steps=math.floor(steps/10) #time expansion factor
    else:
        sample_rate, samples, _, _,steps= AD.spect('Audio_data/'+ file_name);
    start_index=0
    stop_index=int(spect_overlap_window*len(samples)/(steps*spect_window))
    for i in range(steps):
        samples_dummy=samples[start_index:stop_index]
        start_index=stop_index
        stop_index+=int(spect_overlap_window*len(samples)/(steps*spect_window))
        spectros[i]=AD.spect_plot(samples_dummy,sample_rate)
        ctrs, dummy_flag=AD.ROI(spectros[i], kern, X)
        if dummy_flag:
            rectangles[i], regions[i]=AD.ROI2(ctrs, spectros[i])
    rectangles2, regions2=AD.overload(rectangles, regions)
    return(rectangles2, regions2, spectros)
    
def ROI(spect_norm, kern, X):
    #kern: parameters of the kernel size
    len_flag=True
    #binary
    #Conversion to uint8 for contours
    ret,thresh = cv2.threshold(spect_norm,X,256,cv2.THRESH_BINARY)
    #dilation
    kernel = np.ones((kern[0],kern[1]), np.uint8)
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

def overload(rectangles, regions): #deletes entries with ten or more rectangles
    para=AD.set_parameters()
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
    if 'path' in optional:
        plt.savefig(optional['path'])
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

def create_smatrix(rectangles, spectros, num_classes): #creates empty score matrix
    x=len(spectros)
    dummy=np.zeros(x, dtype=np.uint8)
    for i,j in rectangles.items():
        dummy[i]=len(rectangles[i][0,:])
    y=dummy.max()
    s_mat=np.zeros((y, x, num_classes), dtype=np.float)
    return(s_mat)

def calc_smatrix(s_mat, regions, rectangles, templates, num): #Fills s_mat
    s_mat2=s_mat.copy()
    a=len(templates)
    min_freq,max_freq=AD.set_freqthresh(num)
    dummy=np.zeros(a, dtype=np.float)
    for i,d in regions.items():
        for j,d in regions[i].items():
            for k in range(a):
                dummy[k]=AD.compare_img(regions[i][j], templates[k], rectangles[i][:, j], min_freq, max_freq)
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
    para=AD.set_parameters();
    thresh=para[3]
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

def loop_res(rectangles, spectros, regions, templates): 
    s_mat=AD.create_smatrix(rectangles, spectros, len(templates))
    for i in range(len(templates)):
        s_mat=AD.calc_smatrix(s_mat, regions, rectangles, templates[i], i)
    c_mat=AD.create_cmatrix(rectangles, spectros)
    c_mat=AD.calc_cmatrix(c_mat, s_mat)
    res=AD.calc_result(c_mat, len(templates))
    return(res, c_mat, s_mat)

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

def loop_full(file_name):
    rectangles, regions, spectros=AD.spect_loop(file_name)
    templates=AD.create_template_set()
    res, _, _=AD.loop_res(rectangles, spectros, regions, templates)
    return(res)

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

def calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, weight):
    w1,w2,w3,w4,w5,w6=AD.set_weights(weight)
    dist_mat=(w1*sim_mat1)+(w2*sim_mat2)+(w3*sim_mat3)+(w4*sim_mat4)+(w4*sim_mat4)+(w4*sim_mat4)
    return(dist_mat)

def set_weights(weight):
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

def run_MDS(weight):
    rectangles_final, regions_final=AD.set_templates2()
    sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6=AD.calc_sim_matrix(rectangles_final, regions_final)
    dist_mat=AD.calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, weight)
    pos=AD.calc_pos(dist_mat)
    AD.plot_MDS(pos)
    return()

def run_TSNE(weight):
    rectangles_final, regions_final=AD.set_templates2()
    sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6=AD.calc_sim_matrix(rectangles_final, regions_final)
    dist_mat=AD.calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, weight)
    pos=AD.calc_pos_TSNE(dist_mat)
    AD.plot_MDS(pos)
    return()

def calc_features(rectangles, regions, templates, num_reg, list_bats, num_total):
    features=np.zeros((num_reg, len(templates)+5))
    features_freq=np.zeros((num_reg, 5)) #unscaled freq info
    count=0
    features_key={}
    for i,d in regions.items():
        for j,d in regions[i].items():
            features_key[count]=(i,j)
            features[count, 0]=rectangles[i][3,j] #freq range
            features[count, 1]=rectangles[i][1,j] #min freq
            features[count, 2]=rectangles[i][1,j]+rectangles[i][3,j] #max freq
            features[count, 3]=rectangles[i][1,j]+rectangles[i][3,j]/2 #av freq
            features[count, 4]=rectangles[i][2,j] #duration    
            for k in range(len(templates)):
                features[count, k+5]=AD.compare_img2(regions[i][j], templates[k])
            features_freq[count]=features[count, :5]
            count+=1
    #Feature scaling, half of the clustering is based on freq and time information
    features[:,0]=(num_total/5)*(features[:,0]-features[:,0].min())/(features[:,0].max()-features[:,0].min())
    features[:,1]=(num_total/5)*(features[:,1]-features[:,1].min())/(features[:,1].max()-features[:,1].min())
    features[:,2]=(num_total/5)*(features[:,2]-features[:,2].min())/(features[:,2].max()-features[:,2].min())
    features[:,3]=(num_total/5)*(features[:,3]-features[:,3].min())/(features[:,3].max()-features[:,3].min())
    features[:,4]=(num_total/5)*(features[:,4]-features[:,4].min())/(features[:,4].max()-features[:,4].min())
    return(features, features_key, features_freq)

def calc_num_regions(regions):
    num_reg=0
    for i,d in regions.items():
        for j,d in regions[i].items():
            num_reg+=1
    return(num_reg)

def calc_col_labels(features): #based upon maximum ssim
    label_colors={}
    para=AD.set_parameters();
    thresh=para[3]
    for i in range(len(features)):
        dummy_index=features[i,5:].argmax()+5 #index max ssim
        dummy_value=features[i,5:].max() #maximum ssim
        if dummy_value>thresh: #bat
            if 4<dummy_index<44: #ppip, 5-43
               label_colors[i]="#ff0000" #red
            elif 43<dummy_index<61: #eser, 44-60
                label_colors[i]="#008000" #green
            elif 60<dummy_index<67: #mdau, 61-66
                label_colors[i]="#0000ff" #blue
            elif 66<dummy_index<85: #pnat, 67-84
                label_colors[i]="#a52a2a" #brown
            elif 84<dummy_index<91: #nlei, 85-90
                label_colors[i]="#ee82ee" #violet
            else: #temp code
                label_colors[i]="#000000" #black
        else: #noise
            label_colors[i]= "#000000" #black      
    return(label_colors)

def calc_col_labels2(features, features_freq, freq_bats, freq_range_bats, list_bats, colors_bat, num_bats): #based upon percentage scores
    label_colors={}
    per_total={}
    per_total2={}
    para=AD.set_parameters();
    thresh=para[3]
    for i in range(len(features)): #check rows one by one
        count=np.zeros((len(list_bats),)) #counters per
        count2=np.zeros((len(list_bats),)) #counters reg
        per=np.zeros((len(list_bats),)) #percentage scores
        per2=np.zeros((len(list_bats),)) #percentage scores reg
        weight=0
        dummy=(features[i,5:]>thresh) #matching bats
        for j in range(len(dummy)):
            lower_bound=0
            upper_bound=num_bats[0]
            for k in range(len(list_bats)):
                if k==0: #k-1 doesn't exist at first
                    if j<=upper_bound and dummy[j]==True: #match
                        weight=1+((features_freq[i,1]-freq_bats[k]/5)**2)+((features_freq[i,0]-freq_range_bats[k]/5)**2)
                        count[k]+=(1/weight)
                        count2[k]+=1
                else: #every other k
                    lower_bound+=num_bats[k-1]
                    upper_bound+=num_bats[k]
                    if lower_bound<j<=upper_bound and dummy[j]==True: #match
                        weight=1+((features_freq[i,1]-freq_bats[k]/5)**2)+((features_freq[i,0]-freq_range_bats[k]/5)**2)
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

def set_batfreq(rectangles_temp, list_bats, num_bats): #sets the lowest frequency of each bat
    #Change this to use rectangles instead
    freq_bats=[None] *len(list_bats)
    freq_range_bats=[None] *len(list_bats)
    max_index=0
    for i in range(len(list_bats)):
        tot_freq=list()
        tot_freq_range=list()
        for j in range(num_bats[i]):
            tot_freq.append(rectangles_temp[max_index+j][1])
            tot_freq_range.append(rectangles_temp[max_index+j][3])
        max_index+=j #keep counting
        freq_bats[i]=np.median(tot_freq)
        freq_range_bats[i]=np.median(tot_freq_range)
    return(freq_bats, freq_range_bats)

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
    linked = linkage(features, 'average')
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
    f, ax1 = plt.subplots()
    ax1.imshow(spectros[a])
    rect = patches.Rectangle((rectangles[a][0,b],rectangles[a][1,b]),
                                rectangles[a][2,b],rectangles[a][3,b],
                                linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    para=AD.set_parameters();
    min_spec_freq=para[4]
    min_freq=int((rectangles[a][1,b]+min_spec_freq)*0.375)
    max_freq=int((rectangles[a][1,b]+rectangles[a][3,b]+min_spec_freq)*0.375)
    plt.title('%d-%d kHz, timestep: %d' %(min_freq,max_freq, a)) #Show frequency range and time as title
    if 'name' in optional:
        plt.savefig(optional['name'] + '.png')
    else:
        plt.show()
    plt.close()
    return()

def hier_clustering(file_name, freq_bats, freq_range_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp, **optional):
    rectangles, regions, spectros=AD.spect_loop(file_name)
    num_reg=AD.calc_num_regions(regions)
    features, features_key, features_freq=AD.calc_features(rectangles, regions, templates, num_reg, list_bats, num_total)
    col_labels, per_total, per_total2=AD.calc_col_labels2(features, features_freq, freq_bats, freq_range_bats, list_bats, colors_bat, num_bats)
    if 'write' in optional:
        if optional['write']: #true
            AD.plot_dendrogram(features, col_labels, name=file_name)
        else: #false
            AD.plot_dendrogram(features, col_labels)
    else:
        AD.plot_dendrogram(features, col_labels)
    return(col_labels, features_key, rectangles, spectros, per_total, per_total2)

def write_output(list_files, output_file, **optional): #Optional only works on non TE data
    #loading
    freq_bats, freq_range_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp=AD.loading_init(**optional)
    if 'full' in optional:
       if optional['full']: #True
           if 'Audio_data' in optional:
               path=optional['Audio_data']
           else:
               path=AD.set_path()
           path=AD.set_path()
           list_files2=os.listdir(path + '/Audio_data') #write out results for everything
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
    #run clustering and save output    
    for i in range(len(list_files2)):
        col_labels[i], features_key[i], rectangles[i], spectros[i], per_total[i], per_total2[i]=AD.hier_clustering(list_files2[i], freq_bats, freq_range_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp, write=True)
    total_count=np.zeros((len(list_bats), 1), dtype=np.uint8)
    #output file
    f=open(output_file, 'a').close() #create file if it doesn't exist yet 
    open(output_file, 'w').close() #clear file
    f=open(output_file, 'a')
    for k in range(len(list_bats)):
        f.write(str(list_bats[k]) +': ' + '\n'); #name bat
        f.write('\n') #skip line
        for i in range(len(list_files2)):
            f.write(str(i) + ': ' + list_files2[i] + "\n") #name file
            count=0
            for j in range(len(col_labels[i])):
                if col_labels[i][j]==colors_bat[list_bats[k]]:
                    f.write('Time: ' + str(features_key[i][j][0]/10) + ' s, region: ' + str(features_key[i][j][1]) + ', score1: ' + str(int(100000*per_total[i][j][k])) + ' mil, score2: ' + str(int(100*per_total2[i][j][k])) + '% \n');
                    count+=1
                    temp_str=list_bats[k] + '/time_' + str(features_key[i][j][0]/10) + '_region_' + str(features_key[i][j][1]) + '_file_' + str(list_files2[i])
                    show_region2(rectangles[i], spectros[i], features_key[i], j, name=temp_str)
            f.write('Total: ' + str(count) + '\n')
            f.write('\n') #empty line between different files
            total_count[k]+=count
    f.write('Summary: \n')
    for k in range(len(list_bats)):
        f.write(str(list_bats[k]) +': ' + str(total_count[k]) + '\n');
    f.close()
    return()

def create_template(file_name, time, region_num, bat_name, **optional): #creates two templates (image and array)
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD.set_path()
    AD.make_folders(path)
    os.chdir(path)
    list_bats, _=AD.set_batscolor()
    num_bats, _=AD.set_numbats(list_bats, optional)
    if bat_name in list_bats: #bat already exists
        i=list_bats.index(bat_name)
        dummy=str(num_bats[i])
    else: #bat doesn't exist yet, create new folders
        os.makedirs(path + '/Templates_arrays/' + bat_name)
        os.makedirs(path + '/Templates_images/' + bat_name)
        os.makedirs(path + '/Templates_rect/' + bat_name)
        dummy=str(0)
    path_total='Templates_images/' + bat_name + '/' + dummy + '.png'
    rectangles, regions, _=AD.spect_loop(file_name)
    plt.imshow(regions[int(time*10)][region_num])
    plt.savefig(path_total)
    plt.close()
    path_total='Templates_arrays/' + bat_name + '/' + dummy + '.npy'
    path_total_rec='Templates_rect/' + bat_name + '/' + dummy + '.npy'
    np.save(path_total, regions[int(time*10)][region_num])
    np.save(path_total_rec, rectangles[int(time*10)][:, region_num])
    return()

def read_templates(**optional): #reads in templates from the path to the general folder
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD.set_path()
    full_path='' #string will be constructed every step
    list_bats, _=AD.set_batscolor()
    if 'Templates' in optional:
        num_bats, _=AD.set_numbats(list_bats, Templates=optional['Templates'])
    else:
        num_bats, _=AD.set_numbats(list_bats)
    regions={}
    rectangles={}
    count=0
    for i in range(len(list_bats)):
        for j in range(num_bats[i]):
            #Make path to go through each file one by one
            full_path=path+ '/Templates_arrays/' + list_bats[i] + '/' + str(j) +'.npy'
            full_path_rec=path+ '/Templates_rect/' + list_bats[i] + '/' + str(j) +'.npy'
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

def loading_init(**optional): #loads in certain things so they only run once
    if 'Templates' in optional:
        regions_temp, rectangles_temp=AD.read_templates(Templates=optional['Templates'])
        list_bats, colors_bat=AD.set_batscolor(Templates=optional['Templates'])
        num_bats, num_total=AD.set_numbats(list_bats, Templates=optional['Templates'])
    else:
        regions_temp, rectangles_temp=AD.read_templates()
        list_bats, colors_bat=AD.set_batscolor()
        num_bats, num_total=AD.set_numbats(list_bats)
    freq_bats, freq_range_bats=AD.set_batfreq(rectangles_temp, list_bats, num_bats)
    return(freq_bats, freq_range_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp)
    