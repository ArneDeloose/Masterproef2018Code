
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD2_Spectro as AD2
import AD4_SOM as AD4

path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)


freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD1.loading_init()

#test files
name_file='101_B-barbastellus_capt_Hte-Vienne_May2010_M-Barataud'

rectangles, regions, spectros=AD2.spect_loop('Barataud_CD_TimeExpansion10x/'+name_file + '.wav', exp_factor=10)

AD2.show_mregions(rectangles, spectros)



#define eval templates

#bbar: 5 pulses, file 100
bat_list1=(0,3,4,6,9)
bat_list2=(0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/100_alt_capt_B-barbastellus_Charente-Mar_May2002_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'bbar', exp_factor=10, template_type='evaluate')


#evaluation plot
X_final, Y_final, net, D=AD4.evaluation_SOM(dim1=20, dim2=20, export='Evaluation_bbar (no_eval)')
