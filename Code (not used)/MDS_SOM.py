import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD4_SOM as AD4
import AD5_MDS as AD5

path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')

net, raw_data=AD4.fit_SOM(list_files)

U=AD4.calc_Umat(net)
import matplotlib.pyplot as plt
f, ax1 = plt.subplots()
plt.imshow(U)
plt.colorbar()
plt.show()
f.savefig('Umat.jpg', format='jpg', dpi=1200)
plt.close()


#plot MDS for full data
net_features=AD4.calc_net_features(net)
D=AD5.calc_dist_matrix(net_features, 1, raw_data=raw_data)
pos=AD5.calc_pos(D)
AD5.plot_MDS2(pos, 5, 5)

score=AD4.calc_BMU_scores(raw_data, net)


net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2=AD4.calc_output(list_files, net)

full_region, full_rectangle, full_spectro, full_name=AD4.rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net)

AD4.plot_region_neuron(full_region, full_rectangle, full_spectro, full_name, 1, 2, 3)

#####


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=np.transpose(raw_data)


#pd.scatter_matrix(data, alpha = 0.3, figsize = (14,10), diagonal='kde');
#plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=6).fit(data)
pca_samples = pca.transform(data)

keys=list(range(93))

keys[0]='freq range'
keys[1]='min freq'
keys[2]='max freq'
keys[3]='av freq'
keys[4]='duration'
keys[5]='peak freq T'
keys[6]='peak freq F'

def pca_results(data, pca):
    
    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = keys)
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance']) 
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Feature Weights") 
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios# 
    for i, ev in enumerate(pca.explained_variance_ratio_): 
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

#pca_results = pca_results(data, pca)

# creating a biplot

pca = PCA(n_components=2).fit(data)
reduced_data = pca.transform(data)
pca_samples = pca.transform(data)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

def biplot(data, reduced_data, pca):
    
    fig, ax = plt.subplots(figsize = (14,8))
    
    # scatterplot of the reduced data 
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # using scaling factors to make the arrows
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, keys[i], color='black', ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax

#biplot(data, reduced_data, pca)

#####
#reduced PCA

data_red=data[:, 0:7]
keys=list(range(7))

keys[0]='freq range'
keys[1]='min freq'
keys[2]='max freq'
keys[3]='av freq'
keys[4]='duration'
keys[5]='peak freq T'
keys[6]='peak freq F'

pca = PCA(n_components=6).fit(data_red)
pca_samples = pca.transform(data_red)
pca_results = pca_results(data, pca)

pca = PCA(n_components=2).fit(data_red)
reduced_data = pca.transform(data_red)
pca_samples = pca.transform(data_red)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

biplot(data_red, reduced_data, pca)

