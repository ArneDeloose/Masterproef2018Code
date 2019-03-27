#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

#MDS
AD.run_MDS(0)
#run MDS with single variable
for i in range(1,7):
    AD.run_MDS(i)

#TSNE
AD.run_TSNE(0)
#run MDS with single variable
for i in range(1,7):
    AD.run_TSNE(i)



#Support vector classification
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X, y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([[-0.8, -1]]))
 

#Self-organising map

#MDS without templates
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name1='ppip-1µl1µA044_AAT.wav' #training set
file_name2='ppip-1µl1µA044_ABF.wav' #training set
file_name3='eser-1µl1µA030_ACH.wav' #different bat
file_name4='noise-1µl1µA037_AAB.wav' #noise data
file_name5='eser-1_ppip-2µl1µA043_AEI.wav'


rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)
rectangles4, regions4, spectros4=AD.spect_loop(file_name4)
rectangles5, regions5, spectros5=AD.spect_loop(file_name5)

regions=regions5.copy()

sim_mat=np.ones((10*len(spectros5),10*len(spectros5)))

#there are never more than ten regions
#Start with ones so distance becomes zero if not filled in
#sim_mat[a,b]=distance between region [a/10][remainder a/10] and...
    
#compare_img is non-symmetrical, 
#compare_img2 takes the biggest image and converts the smallest one to that

for i,j in regions.items():
    for a,b in regions[i].items():
        for k,l in regions.items():
            for c,d in regions[k].items():
                sim_mat[10*i+a,10*k+c]=AD.compare_img2(regions[i][a], regions[k][c])

dist_mat=1-sim_mat

seed = np.random.RandomState(seed=3)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(dist_mat).embedding_


s = 10
plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')

templates=AD.create_template_set()
res, c_mat, s_mat=AD.loop_res(rectangles5, spectros5, regions5, templates)

plt.show()



