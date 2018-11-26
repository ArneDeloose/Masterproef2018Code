#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

import matplotlib.pyplot as plt
import numpy as np

_, _, _, _, list_bats, colors_bat, num_bats, num_total, regions, rectangles=AD.loading_init()

path=AD.set_path()
count=0
for i in range(len(list_bats)):
    for j in range(num_bats[i]):
        hash_image=hash(str(regions[count]))
        hash_rect=hash(str(rectangles[count]))
        path_image=path + '/Templates_images/' + list_bats[i] + '/' + str(hash_image) + '.png'
        plt.imshow(regions[count])
        plt.savefig(path_image)
        plt.close()
        path_array=path + '/Templates_arrays/' + list_bats[i] + '/' + str(hash_image) + '.npy'
        path_rect=path + '/Templates_rect/' + list_bats[i] + '/' + str(hash_rect) + '.npy'
        np.save(path_array, regions[count])
        np.save(path_rect, rectangles[count])
        count+=1
