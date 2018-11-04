#%matplotlib qt #plot in seperate window
import time

start=time.clock()

#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

#files
#file_name1='ppip-1µl1µA044_ABF.wav' #ppip
#file_name2='eser-1µl1µA030_ACH.wav' #eser
#file_name3='noise-1µl1µA037_AAB.wav' #noise data
file_name1='eser-1_ppip-2µl1µA043_AEI.wav'
file_name2='eser-1_ppip-2µl1µA048_AFT.wav'
file_name3='ppip-1µl1µB011_ABJ.wav'
file_name4='ppip-1µl1µA045_AAS.wav'
file_name5='mdau-1µl1µA052_AJP.wav'
file_name6='pnat-1µl1µA036_AGJ.wav'
file_name7='A027_AFO.wav'
file_name8='A012_AMH.wav'
file_name9='A051_AMF.wav'
file_name10='A052_ALB.wav'



rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)
rectangles4, regions4, spectros4=AD.spect_loop(file_name4)
rectangles5, regions5, spectros5=AD.spect_loop(file_name5)
rectangles6, regions6, spectros6=AD.spect_loop(file_name6)
rectangles7, regions7, spectros7=AD.spect_loop(file_name7)
rectangles8, regions8, spectros8=AD.spect_loop(file_name8)
rectangles9, regions9, spectros9=AD.spect_loop(file_name9)
rectangles10, regions10, spectros10=AD.spect_loop(file_name10)

num_reg1=AD.calc_num_regions(regions1)
num_reg2=AD.calc_num_regions(regions2)
num_reg3=AD.calc_num_regions(regions3)
num_reg4=AD.calc_num_regions(regions4)
num_reg5=AD.calc_num_regions(regions5)
num_reg6=AD.calc_num_regions(regions6)
num_reg7=AD.calc_num_regions(regions7)
num_reg8=AD.calc_num_regions(regions8)
num_reg9=AD.calc_num_regions(regions9)
num_reg10=AD.calc_num_regions(regions10)

_, templates=AD.set_templates2()

features1, features_key1, features_freq1=AD.calc_features(rectangles1, regions1, templates, num_reg1)
features2, features_key2, features_freq2=AD.calc_features(rectangles2, regions2, templates, num_reg2)
features3, features_key3, features_freq3=AD.calc_features(rectangles3, regions3, templates, num_reg3)
features4, features_key4, features_freq4=AD.calc_features(rectangles4, regions4, templates, num_reg4)
features5, features_key5, features_freq5=AD.calc_features(rectangles5, regions5, templates, num_reg5)
features6, features_key6, features_freq6=AD.calc_features(rectangles6, regions6, templates, num_reg6)
features7, features_key7, features_freq7=AD.calc_features(rectangles7, regions7, templates, num_reg7)
features8, features_key8, features_freq8=AD.calc_features(rectangles8, regions8, templates, num_reg8)
features9, features_key9, features_freq9=AD.calc_features(rectangles9, regions9, templates, num_reg9)
features10, features_key10, features_freq10=AD.calc_features(rectangles10, regions10, templates, num_reg10)

col_labels1=AD.calc_col_labels2(features1, features_freq1)
col_labels2=AD.calc_col_labels2(features2, features_freq2)
col_labels3=AD.calc_col_labels2(features3, features_freq3)
col_labels4=AD.calc_col_labels2(features4, features_freq4)
col_labels5=AD.calc_col_labels2(features5, features_freq5)
col_labels6=AD.calc_col_labels2(features6, features_freq6)
col_labels7=AD.calc_col_labels2(features7, features_freq7)
col_labels8=AD.calc_col_labels2(features8, features_freq8)
col_labels9=AD.calc_col_labels2(features9, features_freq9)
col_labels10=AD.calc_col_labels2(features10, features_freq10)



#%matplotlib qt #plot in seperate window
#%matplotlib inline

AD.plot_dendrogram(features1, col_labels1)
AD.plot_dendrogram(features2, col_labels2)
AD.plot_dendrogram(features3, col_labels3)
AD.plot_dendrogram(features4, col_labels4)
AD.plot_dendrogram(features5, col_labels5)
AD.plot_dendrogram(features6, col_labels6)
AD.plot_dendrogram(features7, col_labels7)
AD.plot_dendrogram(features8, col_labels8)
AD.plot_dendrogram(features9, col_labels9)
AD.plot_dendrogram(features10, col_labels10)


#AD.show_region2(rectangles1, spectros1, features_key1, 5)


end=time.clock()

runtime=end-start
#runtime: 181 sec, analysis 60 sec

