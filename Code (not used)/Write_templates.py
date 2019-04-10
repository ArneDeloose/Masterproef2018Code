import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD2_Spectro as AD2
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

list_bats=['']
AD1.make_folders(path, list_bats)

#plot
rectangles, regions, spectros=AD2.spect_loop('ppip-1µl1µA044_AAT.wav')
AD2.show_mregions(rectangles, spectros)
#AD2.show_region(rectangles, spectros, 2, export='overlap_figure_a')

#define templates

ppip_list1=(2,4,6,8,13,20,25,33)
ppip_list2=(0,0,0,0,0,0,0,0)
for i in range(len(ppip_list1)):
    AD2.create_template('ppip-1µl1µA044_AAT.wav', ppip_list1[i], ppip_list2[i], 'ppip')

eser_list1=(4,7,8,11,12,16,20,25,29,33)
eser_list2=(0,0,0,0,0,0,0,0,0,0)
for i in range(len(eser_list1)):
    AD2.create_template('eser-1µl1µA030_ACH.wav', eser_list1[i], eser_list2[i], 'eser')

pnat_list1=(26,29,34)
pnat_list2=(0,0,0)
for i in range(len(pnat_list1)):
    AD2.create_template('pnat-1_ppip-1µl1µA037_AGQ.wav', pnat_list1[i], pnat_list2[i], 'ppip')

nlei_list1=(15,25,28,35)
nlei_list2=(0,0,0,0)
for i in range(len(nlei_list1)):
    AD2.create_template('nlei-1_ppip-1µl1µA028_AAW.wav', nlei_list1[i], nlei_list2[i], 'nlei')




#templates=AD.read_templates()
#_, regions2=AD.set_templates2()