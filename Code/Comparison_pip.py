#Comparison of two pipistrelle signals and a different signal (eser)

#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name1='ppip-1µl1µA044_AAT.wav' #training set
file_name2='ppip-1µl1µA044_ABF.wav' #test set
file_name3='eser-1µl1µA030_ACH.wav' #different bat
rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)


#Too much noise
#AD.show_region(rectangles2, spectros2, 2)

#Create template set
#import matplotlib.pyplot as plt

#AD.show_region(rectangles1, spectros1, 3)
#plt.imshow(regions1[2][0])

#Pip bat
img1=regions1[0][1]
img2=regions1[1][2]
img3=regions1[2][3]
img4=regions1[3][2]

templates_0={0: img1, 1: img2, 2: img3, 3: img4} #Pip bat

#ppip 
s_mat1=AD.create_smatrix(rectangles1, spectros1, 1)
s_mat1=AD.calc_smatrix(s_mat1, regions1, templates_0, 0)

c_mat1=AD.create_cmatrix(rectangles1, spectros1)
c_mat1=AD.calc_cmatrix(c_mat1, s_mat1)

#ppip test data
s_mat2=AD.create_smatrix(rectangles2, spectros2, 1)
s_mat2=AD.calc_smatrix(s_mat2, regions2, templates_0, 0)

c_mat2=AD.create_cmatrix(rectangles2, spectros2)
c_mat2=AD.calc_cmatrix(c_mat2, s_mat2)

#esper bat test data
s_mat3=AD.create_smatrix(rectangles3, spectros3, 1)
s_mat3=AD.calc_smatrix(s_mat3, regions3, templates_0, 0)

c_mat3=AD.create_cmatrix(rectangles3, spectros3)
c_mat3=AD.calc_cmatrix(c_mat3, s_mat3)