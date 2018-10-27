#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

file_name1='nnoc-1_ppip-1µl1µB012_AAQ.wav'
file_name2='pnat-1µl1µA036_AGJ.wav'
file_name3='mdau-1µl1µA012_AGW.wav'
file_name4='mdau-1µl1µA052_AJP.wav'
file_name5='pnat-1_ppip-1µl1µA037_AGQ.wav'
file_name6='nlei-1_ppip-1µl1µA028_AAW.wav'
file_name7='pnat-1µl1µA036_AGJ.wav'

rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)
rectangles4, regions4, spectros4=AD.spect_loop(file_name4)
rectangles5, regions5, spectros5=AD.spect_loop(file_name5)
rectangles6, regions6, spectros6=AD.spect_loop(file_name6)
rectangles7, regions7, spectros7=AD.spect_loop(file_name7)


AD.show_mregions(rectangles6, spectros6)
