#Make algorithm more robust by taking the average
#of the top three instead of the max

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name='Test.wav'
rectangles, regions, spectros=AD.spect_loop(file_name)

img1=regions[2][0]
img2=regions[2][1]
img3=regions[2][2]
img4=regions[2][3]

templates_0={0: img3, 1: img4}
s_mat=AD.create_smatrix(rectangles, spectros, 1)
s_mat=AD.calc_smatrix(s_mat, regions, templates_0, 0)

thresh=0.75
c_mat=AD.create_cmatrix(rectangles, spectros)
c_mat=AD.calc_cmatrix(c_mat, s_mat, thresh)


AD.show_region(rectangles, spectros, 2)
AD.show_region(rectangles, spectros, 20)
AD.show_region(rectangles, spectros, 31)
AD.show_region(rectangles, spectros, 34)

AD.compare_img_plot(img3,img4)
AD.resize_img_plot(img3,img4)
score=AD.compare_img(img3, img4)

AD.compare_img_plot(img1,img3)
AD.resize_img_plot(img1,img3)
score2=AD.compare_img(img1, img3)

#Plot region
import matplotlib.pyplot as plt
plt.imshow(regions[2][3])