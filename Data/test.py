import os
path=os.getcwd()
os.chdir(path+'/Modules')
import AD1_Loading as AD1
import AD2_Spectro as AD2
import AD3_Features as AD3
import AD4_SOM as AD4
import AD5_MDS as AD5
os.chdir(path)

#SOM
map_name='map1' #change this to the name of the map
net, raw_data=AD1.import_map(map_name) #if the map is in a different location add 'path=...' here
Dim1=net.shape[0] #size of the map is read automatically
Dim2=net.shape[1]

#DML
dml_name='D1' #change this to the name of the matrix
D=AD1.import_dml(dml_name) #if the matrix is in a different location add 'path=...' here

#plot MDS
net_features=AD4.calc_net_features(net)
D=AD5.calc_dist_matrix(net_features, 1, raw_data=raw_data)
pos=AD5.calc_pos(D)
AD5.plot_MDS2(pos, Dim1, Dim2)

Full=False
Folder_name='None'
List_files=['eser-1_ppip-2µl1µA043_AEI.WAV', 'eser-1_ppip-2µl1µA048_AFT.WAV',
            'ppip-1µl1µB011_ABJ.WAV', 'ppip-1µl1µA045_AAS.WAV', 'mdau-1µl1µA052_AJP.WAV']
Subfolders=False

#M: number of regions matching per neuron
net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2=AD4.calc_output(List_files, net, full=Full, folder=Folder_name, subfolders=Subfolders) #add channel here if TE
full_region, full_rectangle, full_spectro, full_name=AD4.rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net, dim1=Dim1, dim2=Dim2)
M=AD4.calc_matching(full_name, dim1=Dim1, dim2=Dim2)



#MDS
net_features=AD4.calc_net_features(net)
D=AD5.calc_dist_matrix(net_features, 1, raw_data=raw_data)
pos=AD5.calc_pos(D)
AD5.plot_MDS2(pos, Dim1, Dim2)

#add 'export' to save the figure
AD4.heatmap_neurons(M)

#add 'export' to save the figure
AD4.plot_U(net)


dim1=0
dim2=0
point=3
max_time=10
min_freq=20
max_freq=80
context=0
FI=0
fig_size=7

max_c=AD4.calc_maxc(full_name, dim1=Dim1, dim2=Dim2)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math


flag=True

if flag:
    plot_flag=True   
    
    time=max_time
    rectangle=full_rectangle[dim1][dim2][point]
    region=full_region[dim1][dim2][point]
    #convert to pixels
    para=AD1.set_parameters()
    #keep in mind a spectrogram starts counting at a specific frequency
    min_spec_freq=para[4]
    time_pixels=int(time/0.32)
    freq_pixels=int((max_freq-min_freq)/0.375)
    region_center=np.zeros((freq_pixels, time_pixels), dtype=np.uint8)
    #location of start and end frequency region above min_freq
    starting_height=(rectangle[1]*0.375)-min_freq+(min_spec_freq*0.375) #in kHz
    #in pixels
    start_h=math.ceil(starting_height/0.375) #start height in pixels (round up) 
    end_h=start_h+region.shape[0] #ending height in pixels
    #time where region ends
    end_time=region.shape[1] #in pixels
    #check boundaries
    
    #starting height
    if starting_height>0: #start of region is above bounding box, fits inside
        start_j=start_h
        offset_j=0
    else: #start of region is below bounding box, start filling region_center at zero
        start_j=0
        offset_j=math.floor(-starting_height/0.375) #amount of pixels below cutoff (round down)
    #ending height
    if end_h<freq_pixels: #end of region is below bounding box, fits inside
        end_j=end_h
    else: #end of region is above bounding box, stop filling at freq_pixels (end of bounding box)
        end_j=freq_pixels
    start_i=0 #always start filling time at zero
    #ending time
    if end_time<time_pixels: #end of region is to the left of bounding box, fits inside
        end_i=end_time
    else: #end of region is to the right of bounding box, stop filling at end of bounding box
        end_i=time_pixels
    
    #coordinates in region
    count_j=0
    count_i=0
    #fill in array
    for i in range(start_i, end_i): #column
        for j in range(start_j, end_j): #row
            region_center[j, i]=region[offset_j+count_j, count_i]
            count_j+=1
        count_j=0 #next column starts
        count_i+=1
 
    #parameters
    para=AD1.set_parameters()
    context_window_freq=para[15]
    
    #size
    fig_size=(fig_size, fig_size)
    
    #create plot
    if context==1 and FI==1:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size,  gridspec_kw = {'width_ratios':[1, 1, 1]})
    if context==1 and FI==0:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size,  gridspec_kw = {'width_ratios':[1, 1]})
    if context==0 and FI==1:
        f, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,  gridspec_kw = {'width_ratios':[1, 1]})
    if context==0 and FI==0:
        f, (ax2) = plt.subplots(1, 1, figsize=fig_size)
     
    #Middle image, always on 
        
    if plot_flag:
        ax2.imshow(region_center, origin='lower', aspect='equal')
        #set up the axis
        plt.draw() #sets up the ticks
        #set labels
        labels_Y = [item.get_text() for item in ax2.get_yticklabels()] #original labels
        labels_y=list() #new labels
        labels_y.append(labels_Y[0])
        for i in range(1, len(labels_Y)):
            labels_y.append(str(round(float((float(labels_Y[i])*0.375)+min_freq), 2)))
        labels_X = [item.get_text() for item in ax2.get_xticklabels()] #original labels
        labels_x=list()
        labels_x.append(labels_X[0])
        for i in range(1, len(labels_X)):
            labels_x.append(str(round(float(float(labels_X[i])/2.34375), 2))) #convert to ms
            ax2.set_xticklabels(labels_x)
            ax2.set_yticklabels(labels_y)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Frequency (kHz)')   
    else: #show empty image
        ax2.imshow(np.zeros((1,1)))
    
    #Left image (full spectro), only of context is on
    
    if context==1 and plot_flag:
        freq1_index=full_rectangle[dim1][dim2][point][1]-context_window_freq
        if freq1_index<0:
            freq1_index=0
        freq2_index=full_rectangle[dim1][dim2][point][1]+full_rectangle[dim1][dim2][point][3]+context_window_freq
        if freq2_index>full_spectro[dim1][dim2][point].shape[0]:
            freq2_index=full_spectro[dim1][dim2][point].shape[0]
        #image
        ax1.imshow(full_spectro[dim1][dim2][point][freq1_index:freq2_index], origin='lower', aspect='auto')
        #rectangle
        rect = patches.Rectangle((full_rectangle[dim1][dim2][point][0], full_rectangle[dim1][dim2][point][1]-freq1_index),
                              full_rectangle[dim1][dim2][point][2], full_rectangle[dim1][dim2][point][3],
                              linewidth=1, edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(rect)
        plt.draw() #sets up the ticks
        #set labels
        labels_Y = [item.get_text() for item in ax1.get_yticklabels()] #original labels
        labels_y=list() #new labels
        labels_y.append(labels_Y[0])
        for i in range(1, len(labels_Y)):
            labels_y.append(str(float((float(labels_Y[i])+freq1_index)*0.375)))
        labels_X = [item.get_text() for item in ax1.get_xticklabels()]
        labels_x=list()
        labels_x.append(labels_X[0])
        for i in range(1, len(labels_X)):
            labels_x.append(str(int(int(labels_X[i])/2.34375))) #convert to ms
        ax1.set_xticklabels(labels_x)
        ax1.set_yticklabels(labels_y)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Frequency (kHz)')   
    
    #Right image, if FI is on
    
    if FI==1 and plot_flag:
        freq1_index=full_rectangle[dim1][dim2][point][1]-context_window_freq
        if freq1_index<0:
            freq1_index=0
        freq2_index=full_rectangle[dim1][dim2][point][1]+full_rectangle[dim1][dim2][point][3]+context_window_freq
        if freq2_index>full_spectro[dim1][dim2][point].shape[0]:
            freq2_index=full_spectro[dim1][dim2][point].shape[0]
        #image
        FI_matrix=AD4.calc_FI_matrix(full_region[dim1][dim2][point])
        ax3.imshow(FI_matrix, origin='lower', aspect='auto')
        plt.draw()
        #labels    
        labels_X = [item.get_text() for item in ax3.get_xticklabels()] #original labels
        labels_x=list() #new labels
        labels_x.append(labels_X[0])
        for i in range(1, len(labels_X)):
            labels_x.append(str(round((float(labels_X[i])+freq1_index+context_window_freq)*0.375, 2)))
        ax3.set_xticklabels(labels_x)
        ax3.set_xlabel('Frequency (kHz)')
        ax3.set_ylabel('Intensity')
    #title plot
    if plot_flag:   
        ax2.set_title(full_name[dim1][dim2][point])
    else:
        ax2.set_title("No more matches. Change sliders.")
    #show plot
    plt.show()
    plt.close()



