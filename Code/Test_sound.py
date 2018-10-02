#Check biggest rectangle size with some examples
#Append all ROIs to this size (add empty space)
#Define 'distance' between pictures
#Use KNN, new data (library) can improve this procedure

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)

import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name='Test.wav'
image_path='C:\\Users\\arne\\Documents\\School\\Thesis\\temp_figure.png'
rectangles, regions=AD.spect_loop(file_name, image_path)

import matplotlib.pyplot as plt
plt.imshow(regions[16][0], cmap='gray')

#highlight=AD.show_last(image_path, rectangles[49])
#import cv2
#cv2.imshow('Test',highlight)
#cv2.waitKey(0)
