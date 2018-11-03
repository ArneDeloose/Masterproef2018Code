#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

#files
file_name1='ppip-1µl1µA044_ABF.wav' #ppip
#file_name2='eser-1µl1µA030_ACH.wav' #eser
#file_name3='noise-1µl1µA037_AAB.wav' #noise data
file_name2='eser-1_ppip-2µl1µA043_AEI.wav'
file_name3='eser-1_ppip-2µl1µA048_AFT.wav'

rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)

num_reg1=AD.calc_num_regions(regions1)
num_reg2=AD.calc_num_regions(regions2)
num_reg3=AD.calc_num_regions(regions3)

_, templates=AD.set_templates2()

features1, features_key1, features_freq1=AD.calc_features(rectangles1, regions1, templates, num_reg1)
features2, features_key2, features_freq2=AD.calc_features(rectangles2, regions2, templates, num_reg2)
features3, features_key3, features_freq3=AD.calc_features(rectangles3, regions3, templates, num_reg3)

col_labels1=AD.calc_col_labels2(features1, features_freq1)
col_labels2=AD.calc_col_labels2(features2, features_freq2)
col_labels3=AD.calc_col_labels2(features3, features_freq3)


#%matplotlib qt #plot in seperate window
#%matplotlib inline

AD.plot_dendrogram(features1, col_labels1)
AD.plot_dendrogram(features2, col_labels2)
AD.plot_dendrogram(features3, col_labels3)

#AD.show_region2(rectangles1, spectros1, features_key1, 5)
