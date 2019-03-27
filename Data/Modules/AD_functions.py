from __future__ import division #changes / to 'true division'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from skimage.measure import compare_ssim as ssim
import os
from scipy.cluster.hierarchy import dendrogram, linkage

#load modules
import AD_functions as AD
import AD1_Loading as AD1
import AD2_Spectro as AD2   
import AD3_Features as AD3 


def compare_img(img1, img2, rectangle, min_freq, max_freq):
    si=(len(img2[0,:]), len(img2))
    img1_new=cv2.resize(img1, si)
    score=ssim(img1_new, img2, multichannel=True)
    if (rectangle[1]+rectangle[3]/2)<min_freq or (rectangle[1]+rectangle[3]/2)>max_freq:
        score=-1 #set to minimum score
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

def calc_col_labels(features, features_freq, freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, **optional): #based upon percentage scores
    label_colors={}
    per_total={}
    per_total2={}
    para=AD1.set_parameters()
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
    rectangles, regions, spectros=AD2.spect_loop(file_name, **optional)
    num_reg=AD3.calc_num_regions(regions)
    features, features_key, features_freq=AD3.calc_features(rectangles, regions, templates, num_reg, list_bats, num_total)
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
    para=AD1.set_parameters()
    if 'spect_window' in optional:
        spect_window=optional['spect_window']
    else:
        spect_window=para[6]
    if 'spect_window_overlap' in optional:
        spect_window_overlap=optional['spect_window_overlap']
    else:
        spect_window_overlap=para[7]
    #loading
    freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, templates, rectangles_temp=AD1.loading_init(**optional)
    if 'full' in optional:
       if optional['full']: #True
           if 'Audio_data' in optional:
               path=optional['Audio_data']
               list_files2=os.listdir(path)
           else:
               path=AD1.set_path()
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
    list_bats, colors_bat=AD1.set_batscolor()
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


def create_template(file_name, timestep, region_num, bat_name, **optional): #creates three templates (image, rectangle and array)
    if 'Templates' in optional:
        path=optional['Templates']
    else:
        path=AD1.set_path()
    AD1.make_folders(path)
    list_bats, _=AD1.set_batscolor()
    num_bats, _=AD1.set_numbats(list_bats, **optional)
    if not bat_name in list_bats: #bat already exists
        os.makedirs(path + '/Templates_arrays/' + bat_name)
        os.makedirs(path + '/Templates_images/' + bat_name)
        os.makedirs(path + '/Templates_rect/' + bat_name)
    rectangles, regions, _=AD2.spect_loop(file_name, **optional)
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
    