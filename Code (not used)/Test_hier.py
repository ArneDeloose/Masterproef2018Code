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

AD.hier_clustering(file_name1)
AD.hier_clustering(file_name2)
AD.hier_clustering(file_name3)
AD.hier_clustering(file_name4)
AD.hier_clustering(file_name5)
AD.hier_clustering(file_name6)
AD.hier_clustering(file_name7)
AD.hier_clustering(file_name8)
AD.hier_clustering(file_name9)
AD.hier_clustering(file_name10)


end=time.clock()

runtime=end-start

