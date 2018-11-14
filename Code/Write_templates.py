import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';
os.chdir(path)
import AD_functions as AD
path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)

ppip_list1=(0,1,2,3,4,5,6,8,9,10,11,12,14,16,17,18,20,22,24,26,28,29,30,31,32,34,35,36,37,38,40,41,42,44,45,47,48,49,52)
ppip_list2=(0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0)
for i in range(len(ppip_list1)):
    AD.create_template('ppip-1µl1µA044_AAT.wav', ppip_list1[i]/10, ppip_list2[i], 'ppip')

eser_list1=(1,3,4,6,11,12,14,15,17,18,19,20,22,23,25,28,41)
eser_list2=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1)
for i in range(len(eser_list1)):
    AD.create_template('eser-1µl1µA030_ACH.wav', eser_list1[i]/10, eser_list2[i], 'eser')

mdau_list1=(4,5,6,14,15,47)
mdau_list2=(0,0,0,0,5,0)
for i in range(len(mdau_list1)):
    AD.create_template('mdau-1µl1µA012_AGW.wav', mdau_list1[i]/10, mdau_list2[i], 'mdau')

pnat_list1=(2,4,5,6,9,19,20,25,26,29,30,31,32,34,35,37,38,40)
pnat_list2=(0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0)
for i in range(len(pnat_list1)):
    AD.create_template('pnat-1_ppip-1µl1µA037_AGQ.wav', pnat_list1[i]/10, pnat_list2[i], 'pnat')

nlei_list1=(5,7,9,10,13,14)
nlei_list2=(0,0,0,0,0,0)
for i in range(len(nlei_list1)):
    AD.create_template('nlei-1_ppip-1µl1µA028_AAW.wav', nlei_list1[i]/10, nlei_list2[i], 'nlei')

#templates=AD.read_templates()
#_, regions2=AD.set_templates2()