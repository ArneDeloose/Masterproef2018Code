#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

file_name1='eser-1µl1µA030_ACH.wav'
file_name2='eser-1_ppip-2µl1µA043_AEI.wav'
file_name3='eser-1_ppip-2µl1µA048_AFT.wav'

rectangles1, regions1, spectros1=AD.spect_loop(file_name1)
rectangles2, regions2, spectros2=AD.spect_loop(file_name2)
rectangles3, regions3, spectros3=AD.spect_loop(file_name3)


#AD.show_mregions(rectangles1, spectros1)

templates=AD.create_template_set()

#ppip 
res1, c_mat1, s_mat1=AD.loop_res(rectangles1, spectros1, regions1, templates)
res2, c_mat2, s_mat2=AD.loop_res(rectangles2, spectros2, regions2, templates)
res3, c_mat3, s_mat3=AD.loop_res(rectangles3, spectros3, regions3, templates)

#Results: (old: 45;18 templates)
#167;0;18
#193;25;24
#91;17;1


#Results: (new: 39;17 templates)
#58;0;17
#56;31;16
#37;20;1