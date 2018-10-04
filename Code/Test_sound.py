#Check biggest rectangle size with some examples
#Append all ROIs to this size (add empty space)
#Define 'distance' between pictures
#Use KNN, new data (library) can improve this procedure
#Set threshold based on average of an image?

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name='Test.wav'
rectangles, regions, spectros=AD.spect_loop(file_name)
AD.show_region(rectangles, spectros, 2)
AD.show_region(rectangles, spectros, 20)
AD.show_region(rectangles, spectros, 31)
AD.show_region(rectangles, spectros, 34)

