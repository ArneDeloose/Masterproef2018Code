import time

start=time.clock()

#init
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')
list_bats=('ppip', 'eser', 'mdau', 'pnat', 'nlei')
colors_bat={'ppip': "#ff0000", 'eser': "#008000", 'mdau': "#0000ff", 
            'pnat': "#a52a2a", 'nlei': "#ee82ee"} #dictionary linking bats to colors

AD.write_output(list_files, list_bats, colors_bat, 'results.txt')

end=time.clock()

runtime=end-start