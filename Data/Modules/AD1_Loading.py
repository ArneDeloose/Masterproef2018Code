#Module which governs the reading and loading parameters and files. 
#This module also governs the creation of necessary folders

#Load packages
from __future__ import division #changes / to 'true division'
import numpy as np
import math
import os

#load modules
import AD1_Loading as AD1

#loads in certain important parameters so they only need to run once
def loading_init(**optional): 
    regions_temp, rectangles_temp=AD1.read_templates(**optional)
    list_bats, colors_bat=AD1.set_batscolor(**optional)
    num_bats, num_total=AD1.set_numbats(list_bats, **optional)
    freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats=AD1.set_batfreq(rectangles_temp, regions_temp, list_bats, num_bats)
    return(freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp)

#Loads in parameters from the 'parameters.txt' file
def set_parameters():
    #read in the file
    path=AD1.set_path()
    f=open(path+'\parameters.txt', 'r')
    a=f.readlines()
    words=[None]*len(a)
    i=0 #counter
    for line in a:
        words[i]=line.split(';') #save text in 'words' array
        i+=1
    #Spectrogram and ROI
    binary_thresh=int(words[1][0])
    spec_min=int(words[2][0])
    spec_max=int(words[3][0])
    spect_window=int(words[4][0])
    spect_window_overlap=int(words[5][0])
    max_roi=int(words[6][0])
    kern=int(words[7][0]) #window for roi
    #SOM
    network_dim1=int(words[9][0])
    network_dim2=int(words[10][0])
    n_iter = int(words[11][0])
    init_learning_rate = float(words[12][0])
    context_window=int(words[13][0])
    context_window_freq_dummy=int(words[14][0])
    #Hierarchical clustering   
    thresh=float(words[16][0])
    w_1=float(words[17][0])
    w_2=float(words[18][0])
    w_3=float(words[19][0])
    w_4=float(words[20][0])
    #Convert parameters
    w_impor=(w_1, w_2, w_3, w_4)
    X=int(binary_thresh*255/100) #Threshold for noise binary image
    min_spec_freq=int(spec_min/0.375) #freq to pixels
    max_spec_freq=int(spec_max/0.375) #freq to pixels
    network_dim = (network_dim1, network_dim2)
    context_window_freq=int(context_window_freq_dummy/0.375)
    #Extra parameters for SOM
    normalise_data = False
    normalise_by_column = False
    fig_size=(10,10)
    #Put everything in one variable
    para=(X, kern, max_roi, thresh, min_spec_freq, max_spec_freq, spect_window, spect_window_overlap, w_impor, \
          network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column, context_window, context_window_freq, fig_size, \
          spec_min, spec_max, binary_thresh)
    return(para)

#Set path to the current working directory
def set_path():
    path=os.getcwd()
    return(path)

#reads in templates from the path to the general folder
def read_templates(**optional): 
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD1.set_path()
    full_path='' #string will be constructed every step
    full_path_rec=''
    list_bats, _=AD1.set_batscolor()
    if 'Templates' in optional:
        num_bats, _=AD1.set_numbats(list_bats, Templates=optional['Templates'])
    else:
        num_bats, _=AD1.set_numbats(list_bats)
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

#makes necessary template folders if they don't exist yet
def make_folders(path): 
    os.chdir(path)
    if not os.path.exists('Templates_arrays'):
        os.makedirs('Templates_arrays')
    if not os.path.exists('Templates_images'):
        os.makedirs('Templates_images')
    if not os.path.exists('Templates_rect'):
        os.makedirs('Templates_rect')
    return()

#imports a SOM
def import_map(map_name, **optional):
    if 'path' in optional:
        path=optional['path']
    else:
        path=AD1.set_path()
    net=np.load(path+ '/' + map_name+ '.npy')
    raw_data=np.load(path+ '/' + map_name+ '_data.npy')
    return(net, raw_data)

#imports a DML matrix
def import_dml(dml, **optional):
    if 'path' in optional:
        path=optional['path']
    else:
        path=AD1.set_path()
    D=np.load(path+ '/' + dml+ '.npy')
    return(D)

def set_numbats(list_bats, **optional): #sets the number of templates per bat
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD1.set_path()
    AD1.make_folders(path)
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
        path=AD1.set_path()
    path=AD1.set_path()
    colors_bat={}
    list_bats=os.listdir(path + '/Templates_arrays')
    colors=("#ff0000", "#008000", "#0000ff", "#a52a2a", "#ee82ee", 
            "#f0f8ff", "#faebd7", "#f0ffff", "#006400", "#ffa500",
            "#ffff00", "#40e0d0", "#4b0082", "#ff00ff", "#ffd700")
    for i in range(len(list_bats)):
        colors_bat[list_bats[i]]=colors[i]
    return(list_bats, colors_bat)

#prints out which features are what, useful to make a correlation plot
def print_features(**optional):
    list_bats, colors_bat=AD1.set_batscolor(**optional)
    num_bats, num_total=AD1.set_numbats(list_bats, **optional)
    a=6
    print('Frequency: 0-'+str(a))
    for i in range(len(list_bats)):
        a+=1
        print(list_bats[i] + ': ' + str(a) + '-' + str(a+num_bats[i]))
        a+=num_bats[i]
    return()
