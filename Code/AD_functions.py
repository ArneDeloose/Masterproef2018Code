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

def set_parameters():
    X=25 #Threshold for noise binary image
    kern=[3,3] #window for roi
    thresh=0.6 #Threshold for ssim classification
    max_roi=10 #Maximum number of regions in a single spectrogram
    min_spec_freq=53 #20 kHz, restriction spectrogram
    max_spec_freq=214 #80.25 kHz
    #1 point= 0.58 ms
    #1 point= 375 Hz
    return(X, kern, thresh, max_roi, min_spec_freq, max_spec_freq)
    
def set_freqthresh(num_class): #frequency number depends on minimum frequency used for the spectrum
    _,_,_,_,min_spec_freq,max_spec_freq=AD.set_parameters()
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
    _,_,_,_,min_spec_freq,max_spec_freq=AD.set_parameters()
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

def spect_loop(file_name): #hybrid code, one plot for 100 ms
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    X, kern, _, _,_,_=AD.set_parameters()
    sample_rate, samples, t, total_time,steps, microsteps= AD.spect(file_name);
    rectangles={};
    regions={};
    spectros={};
    for i in range(steps):
        for j in range(10):
            if j%2==0: #Even number, make a 200 ms plot
                samples_dummy=samples[int(i*sample_rate+sample_rate*j/10):int(i*sample_rate+sample_rate*(j+2)/10)]
                temp_spect=AD.spect_plot(samples_dummy,sample_rate)
            else: #odd number, assign spect_norm for j and j-1
                #j-1
                spect_norm=temp_spect[:, 0:171]# first half
                ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
                if dummy_flag:
                    rectangles[i*10+(j-1)], regions[i*10+(j-1)]=AD.ROI2(ctrs, spect_norm)
                spectros[i*10+(j-1)]=spect_norm
                #j
                spect_norm=temp_spect[:, 172:] #second half
                ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
                if dummy_flag:
                    rectangles[i*10+j], regions[i*10+j]=AD.ROI2(ctrs, spect_norm)
                spectros[i*10+j]=spect_norm
    for j in range(microsteps):
        if j%2==0: #Even number, make a 200 ms plot
            samples_dummy=samples[int(i*sample_rate+sample_rate*j/10):int(i*sample_rate+sample_rate*(j+2)/10)]
            temp_spect=AD.spect_plot(samples_dummy,sample_rate)
        else: #odd number, assign spect_norm for j and j-1
            #j-1
            spect_norm=temp_spect[:, 0:171]# first half
            ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
            if dummy_flag:
                rectangles[(i+1)*10+(j-1)], regions[(i+1)*10+(j-1)]=AD.ROI2(ctrs, spect_norm)
            spectros[(i+1)*10+(j-1)]=spect_norm
            #j
            spect_norm=temp_spect[:, 172:] #second half
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
    _, _, _, max_roi, _,_=set_parameters()
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
    _, _, thresh, _,_,_=set_parameters();
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

def create_template_set(): #temp function storing a template set
    file_name1='ppip-1µl1µA044_AAT.wav' #ppip set
    file_name2='eser-1µl1µA030_ACH.wav' #eser set
    _, regions1, _=AD.spect_loop(file_name1)
    _, regions2, _=AD.spect_loop(file_name2)
    #File 1
    img1=regions1[0][0]
    img2=regions1[1][0]
    img3=regions1[2][0]
    img4=regions1[3][0]
    img5=regions1[4][0]
    img6=regions1[5][0]
    img7=regions1[6][0]
    img8=regions1[8][0]
    img9=regions1[9][0]
    img10=regions1[10][1]    
    img11=regions1[11][1]
    img12=regions1[12][0]
    img13=regions1[14][0]
    img14=regions1[16][0]
    img15=regions1[17][0]
    img16=regions1[18][0]
    img17=regions1[20][0]
    img18=regions1[22][0]
    img19=regions1[24][0]
    img20=regions1[26][0]
    img21=regions1[28][0]
    img22=regions1[29][0]
    img23=regions1[30][0]
    img24=regions1[31][0]
    img25=regions1[32][0]   
    img26=regions1[34][0]
    img27=regions1[35][0]
    img28=regions1[36][0]
    img29=regions1[37][0]
    img30=regions1[38][0]
    img31=regions1[40][0]
    img32=regions1[41][0]   
    img33=regions1[42][0]
    img34=regions1[44][0]
    img35=regions1[45][0]
    img36=regions1[47][1]
    img37=regions1[48][1]
    img38=regions1[49][0]
    img39=regions1[52][0]
    
    #File 2
    img40=regions2[1][0]
    img41=regions2[3][0]
    img42=regions2[4][0]
    img43=regions2[6][0]
    img44=regions2[11][0]
    img45=regions2[12][0]
    img46=regions2[14][0]
    img47=regions2[15][0]
    img48=regions2[17][0]
    img49=regions2[18][0]
    img50=regions2[19][0]
    img51=regions2[20][0]
    img52=regions2[22][0]
    img53=regions2[23][0]
    img54=regions2[25][0]
    img55=regions2[28][1]
    img56=regions2[41][1]
    
    templates_0={0: img1, 1: img2, 2: img3, 3: img4,
             4: img5, 5: img6, 6: img7, 7: img8,
             8: img9, 9: img10, 10: img11, 11: img12,
             12: img13, 13: img14, 14: img15, 15: img16,
             16: img17, 17: img18, 18: img19, 19: img20,
             20: img21, 21: img22, 22: img23, 23: img24,
             24: img25, 25: img26, 26: img27, 27: img28,
             28: img29, 29: img30, 30: img31, 31: img32,
             32: img33, 33: img34, 34: img35, 35: img36,
             36: img37, 37: img38, 38: img39}
    templates_1={0: img40, 1: img41, 2: img42, 3: img43,
             4: img44, 5: img45, 6: img46, 7: img47,
             8: img48, 9: img49, 10: img50, 11: img51,
             12: img52, 13: img53, 14: img54, 15: img55,
             16: img56}
    templates={0: templates_0, 1: templates_1}
    return(templates)

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

def calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6):
    w1,w2,w3,w4,w5,w6=AD.set_weights()
    dist_mat=(w1*sim_mat1)+(w2*sim_mat2)+(w3*sim_mat3)+(w4*sim_mat4)+(w4*sim_mat4)+(w4*sim_mat4)
    return(dist_mat)

def set_weights():
    w1=1
    w2=1/144
    w3=1/400
    w4=1/400
    w5=1/300
    w6=1/20
    return(w1,w2,w3,w4,w5,w6)
    
def calc_pos(dist_mat):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(dist_mat).embedding_
    return(pos)

def plot_MDS(pos):
    s = 10
    plot1=plt.scatter(pos[0:38, 0], pos[0:38, 1], color='turquoise', s=s, lw=0, label='ppip')
    plot2=plt.scatter(pos[39:55, 0], pos[39:55, 1], color='red', s=s, lw=0, label='eser')
    plt.legend(handles=[plot1,plot2])
    plt.show()
    return()

def set_templates2():
    path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
    os.chdir(path)    
    file_name1='ppip-1µl1µA044_AAT.wav' #ppip set
    file_name2='eser-1µl1µA030_ACH.wav' #eser set
    rectangles1, regions1, _=AD.spect_loop(file_name1)
    rectangles2, regions2, _=AD.spect_loop(file_name2)
    rectangles_final=np.zeros((4,0))
    #File 1
    img1=regions1[0][0]
    img2=regions1[1][0]
    img3=regions1[2][0]
    img4=regions1[3][0]
    img5=regions1[4][0]
    img6=regions1[5][0]
    img7=regions1[6][0]
    img8=regions1[8][0]
    img9=regions1[9][0]
    img10=regions1[10][1]    
    img11=regions1[11][1]
    img12=regions1[12][0]
    img13=regions1[14][0]
    img14=regions1[16][0]
    img15=regions1[17][0]
    img16=regions1[18][0]
    img17=regions1[20][0]
    img18=regions1[22][0]
    img19=regions1[24][0]
    img20=regions1[26][0]
    img21=regions1[28][0]
    img22=regions1[29][0]
    img23=regions1[30][0]
    img24=regions1[31][0]
    img25=regions1[32][0]   
    img26=regions1[34][0]
    img27=regions1[35][0]
    img28=regions1[36][0]
    img29=regions1[37][0]
    img30=regions1[38][0]
    img31=regions1[40][0]
    img32=regions1[41][0]   
    img33=regions1[42][0]
    img34=regions1[44][0]
    img35=regions1[45][0]
    img36=regions1[47][1]
    img37=regions1[48][1]
    img38=regions1[49][0]
    img39=regions1[52][0]
    
    rectangles_final=np.c_[rectangles_final, rectangles1[0][:,0], rectangles1[1][:,0], rectangles1[2][:,0], rectangles1[3][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[4][:,0], rectangles1[5][:,0], rectangles1[6][:,0], rectangles1[8][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[9][:,0], rectangles1[10][:,1], rectangles1[11][:,1], rectangles1[12][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[14][:,0], rectangles1[16][:,0], rectangles1[17][:,0], rectangles1[18][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[20][:,0], rectangles1[22][:,0], rectangles1[24][:,0], rectangles1[26][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[28][:,0], rectangles1[29][:,0], rectangles1[30][:,0], rectangles1[31][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[32][:,0], rectangles1[34][:,0], rectangles1[35][:,0], rectangles1[36][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[37][:,0], rectangles1[38][:,0], rectangles1[40][:,0], rectangles1[41][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles1[42][:,0], rectangles1[44][:,0], rectangles1[45][:,0], rectangles1[47][:,1]]
    rectangles_final=np.c_[rectangles_final, rectangles1[48][:,1], rectangles1[49][:,0], rectangles1[52][:,0]]

    #File 2
    img40=regions2[1][0]
    img41=regions2[3][0]
    img42=regions2[4][0]
    img43=regions2[6][0]
    img44=regions2[11][0]
    img45=regions2[12][0]
    img46=regions2[14][0]
    img47=regions2[15][0]
    img48=regions2[17][0]
    img49=regions2[18][0]
    img50=regions2[19][0]
    img51=regions2[20][0]
    img52=regions2[22][0]
    img53=regions2[23][0]
    img54=regions2[25][0]
    img55=regions2[28][1]
    img56=regions2[41][1]
    
    rectangles_final=np.c_[rectangles_final, rectangles2[1][:,0], rectangles2[3][:,0], rectangles2[4][:,0], rectangles2[6][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles2[11][:,0], rectangles2[12][:,0], rectangles2[14][:,0], rectangles2[15][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles2[17][:,0], rectangles2[18][:,0], rectangles2[19][:,0], rectangles2[20][:,0]]
    rectangles_final=np.c_[rectangles_final, rectangles2[22][:,0], rectangles2[23][:,0], rectangles2[25][:,0], rectangles2[28][:,1]]
    rectangles_final=np.c_[rectangles_final, rectangles2[41][:,1]]
    
    regions_final={0: img1, 1: img2, 2: img3, 3: img4,
             4: img5, 5: img6, 6: img7, 7: img8,
             8: img9, 9: img10, 10: img11, 11: img12,
             12: img13, 13: img14, 14: img15, 15: img16,
             16: img17, 17: img18, 18: img19, 19: img20,
             20: img21, 21: img22, 22: img23, 23: img24,
             24: img25, 25: img26, 26: img27, 27: img28,
             28: img29, 29: img30, 30: img31, 31: img32,
             32: img33, 33: img34, 34: img35, 35: img36,
             36: img37, 37: img38, 38: img39,
             39: img40, 40: img41, 41: img42, 42: img43,
             43: img44, 44: img45, 45: img46, 46: img47,
             47: img48, 48: img49, 49: img50, 50: img51,
             51: img52, 52: img53, 53: img54, 54: img55,
             55: img56}
    return(rectangles_final, regions_final)

def run_MDS():
    rectangles_final, regions_final=AD.set_templates2()
    sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6=AD.calc_sim_matrix(rectangles_final, regions_final)
    dist_mat=AD.calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6)
    pos=AD.calc_pos(dist_mat)
    AD.plot_MDS(pos)
    return()

