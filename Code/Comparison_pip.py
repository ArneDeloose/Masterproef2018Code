#Comparison of pipistrelle signals and different signals (eser and noise)

#Load data
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)
file_name1='ppip-1µl1µA044_AAT.wav' #training set
file_name2='ppip-1µl1µA044_ABF.wav' #training set
file_name3='eser-1µl1µA030_ACH.wav' #different bat
file_name4='noise-1µl1µA037_AAB.wav' #noise data
file_name5='ppip-1µl1µB011_ABJ.wav' #ppip walking
file_name6='ppip-1µl1µA045_AAH.wav' #test data
file_name7='ppip-1µl1µA045_AAM.wav' #test data
file_name8='ppip-1µl1µA045_AAR.wav' #test data
file_name9='ppip-1µl1µA045_AAS.wav' #test data
file_name10='ppip-1µl1µA045_ABA.wav' #test data
file_name11='ppip-1µl1µB011_ABJ.wav' #test data


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
rectangles11, regions11, spectros11=AD.spect_loop(file_name11)

#Show region
#AD.show_mregions(rectangles2, spectros2)
#AD.show_region(rectangles1, spectros1, 3)

#Background bat
#AD.show_region(rectangles7, spectros7, 14)


#Pip bat
templates=AD.create_template_set()

#ppip 
res1, c_mat1, s_mat1=AD.loop_res(rectangles1, spectros1, regions1, templates)
res2, c_mat2, s_mat2=AD.loop_res(rectangles2, spectros2, regions2, templates)
res3, c_mat3, s_mat3=AD.loop_res(rectangles3, spectros3, regions3, templates)
res4, c_mat4, s_mat4=AD.loop_res(rectangles4, spectros4, regions4, templates)
res5, c_mat5, s_mat5=AD.loop_res(rectangles5, spectros5, regions5, templates)
res6, c_mat6, s_mat6=AD.loop_res(rectangles6, spectros6, regions6, templates)
res7, c_mat7, s_mat7=AD.loop_res(rectangles7, spectros7, regions7, templates)
res8, c_mat8, s_mat8=AD.loop_res(rectangles8, spectros8, regions8, templates)
res9, c_mat9, s_mat9=AD.loop_res(rectangles9, spectros9, regions9, templates)
res10, c_mat10, s_mat10=AD.loop_res(rectangles10, spectros10, regions10, templates)
res11, c_mat11, s_mat11=AD.loop_res(rectangles11, spectros11, regions11, templates)


#AD.show_class(1, c_mat1, rectangles1, regions1, spectros1)
#AD.show_class(1, c_mat2, rectangles2, regions2, spectros2)
#AD.show_class(1, c_mat3, rectangles3, regions3, spectros3)

#Results (3 files, 1 test, 20 templates, no frequency requirement):
#res1, training: 44 signals (20 defined), 11 noise
#res2, test: 11 signals, 82 noise
#res3, eser bat: 1 signal (misclassified), 111 noise

#Results (11 files, 1 test data, 45 templates, no freq req)
#res1: 7 noise, 47 signals (45 defined),
#res2: 78; 15 (test data)
#res3: 203; 1 (different bat, size of spectro was changed to min 19.875 kHz to adapt for the frequency)
#res4: 164; 0 (noise data, ssim doesn't go above 0.54)
#res5: 38; 31 (walking)
#65; 13 (hand count: 18(+11) signals)
#72; 22 (hand count: 30(+14) signals)
#70; 17
#74; 18
#59; 21
#38; 31

#24% of signals can be extracted (based on last six observations)
#50% on good data, 20% on noisy data

#results (dual plot method, 39 templates)
#2;41
#42;14
#63;0
#158;0
#36;40
#35;11
#50;18 #background noise at R14 and R15 and other regions
#16;14
#17;19
#48;23
#36;40

#Results (11 files, 2 test data, 45+45 templates (ppip and noise), no freq req)
#8;47;0
#74;15;4
#111;0;1
#119;0;45
#38;31;0
#62;13;3
#70;22;2
#68;17;2
#72;18;2
#59;21;0
#38;31;0