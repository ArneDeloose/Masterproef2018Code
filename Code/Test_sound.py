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

img1=regions[2][0]
img2=regions[2][1]
img3=regions[2][2]
img4=regions[2][3]


AD.compare_img_plot(img3,img4)
AD.resize_img_plot(img3,img4)
score=AD.compare_img(img3, img4)

AD.compare_img_plot(img1,img3)
AD.resize_img_plot(img1,img3)
score2=AD.compare_img(img1, img3)

template=img3
s_mat_init=AD.create_smatrix(rectangles, spectros)
s_mat=AD.calc_smatrix(s_mat_init, regions, template)

import matplotlib.pyplot as plt
f, ax1 = plt.subplots()
ax1.imshow(spectros[51])