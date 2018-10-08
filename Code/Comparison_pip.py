#Comparison of two pipistrelle signals

#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name1='Test.wav' #training set
file_name2='Test.wav' #test set
rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)

#Create template set
AD.show_region(rectangles1, spectros1, 0)

#Pip bat
img1=regions[x][y]
img2=regions[x][y]

#Noise
img3=regions[x][y]
img4=regions[x][y]

templates_0={0: img1, 1: img2} #Pip bat
templates_1={0: img2, 1: img3} #Noise

s_mat=AD.create_smatrix(rectangles2, spectros2, 2)
s_mat=AD.calc_smatrix(s_mat, regions2, templates_0, 0)
s_mat=AD.calc_smatrix(s_mat, regions2, templates_1, 1)

thresh=0.75
c_mat=AD.create_cmatrix(rectangles2, spectros2)
c_mat=AD.calc_cmatrix(c_mat, s_mat, thresh)

#Use same template set on a different bat and see if there is misclassification
