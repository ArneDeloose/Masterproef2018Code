

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





####with templates
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

file_name1='ppip-1µl1µA044_AAT.wav' #ppip set
file_name2='eser-1µl1µA030_ACH.wav' #eser set
_, regions1, _=AD.spect_loop(file_name1)
_, regions2, _=AD.spect_loop(file_name2)
dummy=1
if dummy==1:
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
    
    regions_test={0: img1, 1: img2, 2: img3, 3: img4,
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

sim_mat=np.ones((56,56))

for i,j in regions_test.items():
    for k,l in regions_test.items():
        sim_mat[i,k]=AD.compare_img2(regions_test[i], regions_test[k])

dist_mat=1-sim_mat

seed = np.random.RandomState(seed=3)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(dist_mat).embedding_


s = 10
plt.scatter(pos[0:38, 0], pos[0:38, 1], color='turquoise', s=s, lw=0, label='MDS')
plt.scatter(pos[39:55, 0], pos[39:55, 1], color='red', s=s, lw=0, label='MDS')

