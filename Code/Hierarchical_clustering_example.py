import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

file_name1='eser-1_ppip-2µl1µA043_AEI.wav'
file_name2='ppip-1µl1µA045_AAS.wav'
AD.hier_clustering(file_name1)
AD.hier_clustering(file_name2)