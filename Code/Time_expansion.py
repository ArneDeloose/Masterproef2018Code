#Comparison of pipistrelle signals and different signals (eser and noise)

#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name1='ppip-1µl1µA044_AAT.wav' #normal
file_name2='ppip-1µr10µR05_0364.wav' #time expanded

rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2, channel='r')

#AD.show_mregions(rectangles2, spectros2)
