import time

import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

start=time.clock()

#init


list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'eser-1_ppip-2µl1µA048_AFT.wav',
            'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')

AD.write_output(list_files, 'results.txt')
#AD.write_output(1, 'results2.txt', templates, full=True) #no TE data
#AD.write_output(list_files, 'results3.txt', templates, full=False)



end=time.clock()

runtime=end-start
