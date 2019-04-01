#Module for the fitting and visualization of a self-organizing map
#Also governs the fitting of a distance metric learning

#Load packages
from __future__ import division #changes / to 'true division'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import dml

#load modules
import AD1_Loading as AD1
import AD2_Spectro as AD2   
import AD3_Features as AD3 
import AD4_SOM as AD4 

#Fits a SOM. A list of files must be provided. 
#The optional argument 'full' overrides this list and reads everything in the folder. 
#The optional argument 'features'
#This code puts everything in order and then calls the function 'SOM'.
def fit_SOM(list_files, **optional):
    #Optional arguments
    if 'full' in optional: #read all files in folder
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
    #parameters
    para=AD1.set_parameters()
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
        rectangles1, regions1, spectros1=AD2.spect_loop(list_files2[0])
        num_reg=AD3.calc_num_regions(regions1)
        freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD1.loading_init(**optional)
        features1, _, _=AD3.calc_features(rectangles1, regions1, regions_temp, num_reg, list_bats, num_total)
        raw_data=np.zeros((features1.shape[0], 0))
        raw_data=np.concatenate((raw_data, features1), axis=1)
        #other files
        for i in range(1, len(list_files2)):
            rectangles1, regions1, spectros1=AD2.spect_loop(list_files2[i])
            num_reg=AD3.calc_num_regions(regions1)
            features1, features_key1, features_freq1=AD3.calc_features(rectangles1, regions1, regions_temp, num_reg, list_bats, num_total)
            raw_data=np.concatenate((raw_data, features1), axis=1)
    net=AD4.SOM(raw_data, network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column, **optional)
    return(net, raw_data)

#Iterative process to fit a SOM
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
        bmu, bmu_idx = AD4.find_bmu(t, net, m, D)
        # decay the SOM parameters
        r = AD4.decay_radius(init_radius, i, time_constant)
        l = AD4.decay_learning_rate(init_learning_rate, i, n_iter)
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
                    influence = AD4.calculate_influence(w_dist, r)
                    # now update the neuron's weight using the formula:
                    # new w = old w + (learning rate * influence * delta)
                    # where delta = input vector (t) - old w
                    new_w = w + (l * influence * (t - w))
                    # commit the new weight
                    net[x, y, :] = new_w.reshape(1, m)
    if 'export' in optional:
        path=AD1.set_path()
        np.save(path + '/' + optional['export'] + '.npy', net)
        np.save(path + '/' + optional['export'] + '_data.npy', raw_data)
    return(net)

#finds the bmu for a given datapoint t, a network net, a size m and a dml matrix D
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

#subcodes needed for fitting a SOM
def decay_radius(initial_radius, i, time_constant):
    return (initial_radius * np.exp(-i / time_constant))

def decay_learning_rate(initial_learning_rate, i, n_iter):
    return (initial_learning_rate * np.exp(-i / n_iter))

def calculate_influence(distance, radius):
    return (np.exp(-distance / (2* (radius**2))))

#Calculates the U matrix
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

#Calculates the BMU for every datapoint and stores it into an array
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
        _, bmu_idx=AD4.find_bmu(t, net, m, D)
        score_BMU[i, 0]=bmu_idx[0]
        score_BMU[i, 1]=bmu_idx[1]
    return(score_BMU)

#transforms network features to more suitable form
def calc_net_features(net, **optional): 
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD1.set_parameters()
        network_dim= para[9]
    net_features=np.zeros((net.shape[2], network_dim[0]*network_dim[1]))
    count=0
    for i in range(network_dim[0]):
        for j in range(network_dim[1]):
            net_features[:, count]=net[i, j, :]
            count+=1
    return(net_features)

#plots the U-matrix
def plot_U(net, **optional):
    U=AD4.calc_Umat(net)
    f, ax1 = plt.subplots()
    plt.imshow(U)
    plt.colorbar()
    plt.show()
    if 'export' in optional:
        f.savefig(optional['export']+ '.jpg', format='jpg', dpi=1200)
    plt.close()
    return()

#Plots a heatmap of neurons (matches per neuron)
def heatmap_neurons(M, **optional):
    f, ax1 = plt.subplots()
    plt.imshow(M)
    plt.colorbar()
    plt.show()
    if 'export' in optional:
        f.savefig(optional['export']+ '.jpg', format='jpg', dpi=1200)
    plt.close()
    return()

def calc_output(list_files, net, **optional): #Optional only works on non TE data
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
        rectangles[i], regions[i], spectros[i]=AD2.spect_loop(list_files2[i], **optional)
        num_reg=AD3.calc_num_regions(regions[i])
        features[i], features_key[i], features_freq[i]=AD3.calc_features(rectangles[i], regions[i], templates, num_reg, list_bats, num_total)
        net_label[i]=AD4.calc_BMU_scores(features[i], net, **optional)
    return(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2)   

#Tranforms output into a more suitable form for visualization
def rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net, **optional):
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD1.set_parameters()
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
                        temp_key2=AD4.check_key(regions[k][temp_key[0]], temp_key[1])
                        temp_rectangle[count]=rectangles[k][temp_key[0]][:, temp_key2]
                        temp_spectro[count], extra_time=AD4.calc_context_spec(spectros, k, temp_key)
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

#matches indexes rectangles and regions
def check_key(regions, temp_key): 
    count_dummy=0
    for i in range(temp_key+1):
        if str(i) in regions.keys():
            count_dummy+=1
    return(count_dummy-1)

#Calculates the number of matches per neuron.
def calc_matching(full_name, **optional):
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD1.set_parameters()
        network_dim= para[9]
    M=np.zeros((network_dim[0], network_dim[1]), dtype=np.uint8)
    for i in range(network_dim[0]):
        for j in range(network_dim[1]):
            M[i,j]=len(full_name[i][j])
    return(M)

#Plots regions per neuron    
def plot_region_neuron(full_region, full_rectangle, full_spectro, full_name, dim1, dim2, point, **optional):
    if 'context_window_freq' in optional:
        context_window_freq=optional['context_window_freq']
    else:
        para=AD1.set_parameters()
        context_window_freq=para[15]
    if 'fig_size' in optional:
        fig_size=optional['fig_size']
    else:
        para=AD1.set_parameters()
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
    
    FI_matrix=AD4.calc_FI_matrix(full_region[dim1][dim2][point])
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

#Calculates a context spectrogram
def calc_context_spec(spectros, k, temp_key, **optional): 
    para=AD1.set_parameters()
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

#Calculates a frequency-intensity matrix of a region that can be plotted
def calc_FI_matrix(region):
    FI_matrix=np.zeros((256, region.shape[0]), dtype=np.uint8)
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            dummy=region[i,j] #intensity
            FI_matrix[dummy, i]+=1
    return(FI_matrix[1:, :])

#Calculates max number of matches in a SOM. Needed to set the maximum of a slider
def calc_maxc(full_names, **optional):
    if 'dim1' in optional and 'dim2' in optional:
        network_dim=(optional['dim1'], optional['dim2'])
    else:
        para=AD1.set_parameters()
        network_dim=para[9]
    max_c=0
    for i in range(network_dim[0]):
        for j in range(network_dim[1]):
            temp_c=len(full_names[i][j])
            if temp_c>max_c:
                max_c=temp_c
    return(max_c)

#Fits a dml based on the templates folder and templates_dml folder and 
#Optional argument 'data_X' and 'data_Y' can be used to fit a dml based on custom data 
#Optional argument 'export' is used to save the matrix
def fit_dml(**optional):
    #set parameters
    if 'path' in optional:
        path=optional['path']
    else:
        path=AD1.set_path()
    if 'data_X' in optional and 'data_Y' in optional: #data given
        X=optional['data_X']
        Y=optional['data_Y']
    else: #read in data
        list_bats, _=AD1.set_batscolor(Templates=path) #bat species
        num_bats, num_total=AD1.set_numbats(list_bats, Templates=path) #number of each bat species
        num_bats_dml, num_total_dml=AD1.set_numbats(list_bats, Templates=path+'/Templates_dml')
        #read normal templates
        regions, rectangles=AD1.read_templates(Templates=path)
        #read dml_templates
        regions2, rectangles2=AD1.read_templates(Templates=path+'/Templates_dml')
        #set variables
        templates=regions.copy()
        #combine both rectangles and regions
        for k in range(num_total_dml):
            rectangles[num_total+k]=rectangles2[k]
            regions[num_total+k]=regions2[k]
        
        #calculate features
        features=AD3.calc_features2(rectangles, regions, templates, list_bats, num_total)
        X=np.transpose(features)
        Y=np.zeros((X.shape[1],), dtype=np.uint8)
        #Fill in Y matrix
        count=0
        #go through the templates
        for i in range(len(list_bats)): #i corresponds to a bat species 
            for j in range(num_bats[i]): #number of this type of bat
                Y[count]=i
                count+=1
        #same thing, but for templates_dml
        for i in range(len(list_bats)): #i corresponds to a bat species 
            for j in range(num_bats_dml[i]): #number of this type of bat
                Y[count]=i
                count+=1
    #fit model
    model=dml.anmm.ANMM()
    model.fit(X,Y)
    D=model.transformer()
    #Export D-matrix
    if 'export' in optional:
        np.save(path + '/' + optional['export'] + '.npy', D)
    return(D)



