import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_files=('eser-1_ppip-2µl1µA043_AEI.wav', 'ppip-1µl1µB011_ABJ.wav', 'ppip-1µl1µA045_AAS.wav', 'mdau-1µl1µA052_AJP.wav')

map_name='map1' #change this to the name of the map
net, raw_data=AD.import_map(map_name)
Dim1=net.shape[0]
Dim2=net.shape[1]

net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2=AD.calc_output(list_files, net)
full_region, full_rectangle, full_spectro, full_name=AD.rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net, dim1=Dim1, dim2=Dim2)
M=AD.calc_matching(full_name, dim1=Dim1, dim2=Dim2)

rectangles1, regions1, spectros1=AD.spect_loop('ppip-1µl1µB011_ABJ.wav')

