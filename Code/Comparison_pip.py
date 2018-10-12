#Comparison of two pipistrelle signals and a different signal (eser)

#Add a few more pip signals to test further and use frequency as parameter to determine signal

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

#AD.show_mregions(rectangles1, spectros1)

#Create template set
#import matplotlib.pyplot as plt

#AD.show_region(rectangles1, spectros1, 3)
#plt.imshow(regions1[2][0])

#Pip bat
templates_0=AD.create_template_set(regions1)

#ppip 
res1=AD.loop_res(rectangles1, spectros1, regions1, templates_0)
res2=AD.loop_res(rectangles2, spectros2, regions2, templates_0)
res3=AD.loop_res(rectangles3, spectros3, regions3, templates_0)

#Current results (20 templates, no frequency requirement):
#res1: 44 signals (20 defined), 46 noise
#res2: 11 signals, 128 noise
#res3: 1 signal (misclassified), 72 noise
