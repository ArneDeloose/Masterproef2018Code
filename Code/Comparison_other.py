#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

file_name1='.wav'
file_name2='.wav'
file_name3='.wav'

rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)


AD.show_mregions(rectangles1, spectros1)
