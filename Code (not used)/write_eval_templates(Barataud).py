
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
name_file='080_FM-nasal_B-barbastellus_appr_FME-H1_Hte-Vienne_June2010_M-Barataud'

rectangles, regions, spectros=AD2.spect_loop('Barataud_CD_TimeExpansion10x/'+name_file + '.wav', exp_factor=10)

AD2.show_mregions(rectangles, spectros)


#define eval templates


#ppip: 3 pulses, file 004
bat_list1=(0,7,10)
bat_list2=(0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/004_short-QCF_P-pipistrellus_Hte-Vienne_December2008_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'ppip_eval', exp_factor=10, template_type='evaluate')

#pkuh: 4 pulses, file 006
bat_list1=(0,3,7,10)
bat_list2=(0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/006_short-flat-ended-FM_P-kuhlii_Hte-Vienne_December2008_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'pkuh_eval', exp_factor=10, template_type='evaluate')

#mmyo: 5 pulses, file 007
bat_list1=(2,4,7,10,14)
bat_list2=(0,0,0,0, 0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/007_long-steep-FM_M-myotis_Bourgogne_September2001_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'mmyo_eval', exp_factor=10, template_type='evaluate')

#mdau: 6 pulses, file 008
bat_list1=(0,2,4,6,8,9)
bat_list2=(0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/008_short-steep-FM_M-daubentonii_Hte-Vienne_July2002_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'mdau_eval', exp_factor=10, template_type='evaluate')

#mdau_qcf: 5 pulses, file 009
bat_list1=(1,4,7,12,15)
bat_list2=(0,0,1,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/009_QCF-FM_M-daubentonii_Alpes-Hte-Provence_July2006_P-Favre.wav',
                        bat_list1[i], bat_list2[i], 'mdau_qcf_eval', exp_factor=10, template_type='evaluate')

#msch: 3 pulses, file 015
bat_list1=(1,4,8)
bat_list2=(0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/015_short-flat-ended-FM_ES_M-schreibersii_Herault_November2008_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'msch_eval', exp_factor=10, template_type='evaluate')

#msch_ESsat: 8 pulses, file 017
bat_list1=(1,4,6,8,11,13,15,18)
bat_list2=(0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/017_ES-saturation_M-schreibersii_Corsica_August2001_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'msch_ESsat_eval', exp_factor=10, template_type='evaluate')

#mbec: 2 pulses, file 024
bat_list1=(3,6)
bat_list2=(0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/024_steep-FM_abs_M-bechsteinii_Hte-Vienne_July1999_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'mbec_eval', exp_factor=10, template_type='evaluate')

#mmyo_FM: 2 pulses, file 025
bat_list1=(1,4)
bat_list2=(0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/025_steep-FM_abs_M-myotis_Allier_July1999_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'mmyo_FM_eval', exp_factor=10, template_type='evaluate')

#mdau: 3 pulses, file 026
bat_list1=(1,6,8)
bat_list2=(0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/026_steep-FM_ampl-mod_M-daubentonii_Creuse_May1998_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'mdau_eval', exp_factor=10, template_type='evaluate')

#eser: 9 pulses, file 038
bat_list1=(2,6,14,22,29,37,41,45,52)
bat_list2=(0,0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/038_flat-ended-FM_E-serotinus_Hte-Vienne_August1991_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'eser_eval', exp_factor=10, template_type='evaluate')

#nlei: 5 pulses, file 040
bat_list1=(2,6,10,13,17)
bat_list2=(0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/040_QCF_flat-ended-FM_N-leisleri_approach_Correze_May1995_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'nlei_eval', exp_factor=10, template_type='evaluate')

#tten_nlei_pkuh: 17 pulses, file 051
#bat_list1=(1,3,4,5,6,7,12,16,17,18,20,22,23,23,26,27,43)
#bat_list2=(0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)
#for i in range(len(bat_list1)):
    #AD2.create_template('Barataud_CD_TimeExpansion10x/051_QCF_flat-ended-FM_T-teniotis_N-leisleri_P-kuhlii_Alpes-Mar_August1996_M-Barataud.wav',
                        #bat_list1[i], bat_list2[i], 'tten_eval', exp_factor=10, template_type='evaluate')

#ppyg: 5 pulses, file 052
bat_list1=(0,5,11,13,16)
bat_list2=(0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/052_flat-ended-FM_P-pygmaeus_Lozere_May2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'ppyg_eval', exp_factor=10, template_type='evaluate')

#ppyg: 6 pulses, file 053
bat_list1=(1,3,3,5,11,14)
bat_list2=(0,0,1,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/053_QCF_flat-ended-FM_P-pygmaeus_Lozere_May2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'ppyg_eval', exp_factor=10, template_type='evaluate')

#ppyg: 6 pulses, file 055
bat_list1=(0,2,8,12,14,16)
bat_list2=(0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/055_flat-ended-FM_P-pygmaeus_P-kuhlii_Lozere_May2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'ppyg_eval', exp_factor=10, template_type='evaluate')

#msch_FM: 4 pulses, file 056
bat_list1=(0,2,4,5)
bat_list2=(0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/056_flat-ended-FM_M-schreibersii_Correze_June2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'msch_FM_eval', exp_factor=10, template_type='evaluate')

#msch: 30 pulses, file 057
bat_list1=(1,7,9,11,12,13,16,30,31,34, 36,38,39,40,42,43,48,52,55,60, 63,65,69,76,77,78,80,82,95,100)
bat_list2=(0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/057_flat-ended-FM_M-schreibersii_Correze_June2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'msch_eval', exp_factor=10, template_type='evaluate')

#msch: 11 pulses, file 059
bat_list1=(2,5,8,19,21,23,24,27,30,40,41)
bat_list2=(0,0,0,0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/059_flat-ended-FM_QCF_M-schreibersii_Correze_June2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'msch_eval', exp_factor=10, template_type='evaluate')

#msch: 8 pulses, file 060
bat_list1=(12,13,14,15,18,23,26,27)
bat_list2=(0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/060_flat-ended-FM_M-schreibersii_Correze_June2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'msch_eval', exp_factor=10, template_type='evaluate')

#ppip: 3 pulses, file 061
bat_list1=(1,3,7)
bat_list2=(0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/061_flat-ended-FM_P-pipistrellus_Corsica_July2002_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'ppip_eval', exp_factor=10, template_type='evaluate')

#ppip: 6 pulses, file 062
bat_list1=(1,3,5,7,10,13)
bat_list2=(0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/062_flat-ended-FM_P-pipistrellus_Allier_May2003_S-Giosa.wav',
                        bat_list1[i], bat_list2[i], 'ppip_eval', exp_factor=10, template_type='evaluate')

#ppip: 8 pulses, file 063
bat_list1=(1,4,6,10,12,14,16,19)
bat_list2=(0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/063_flat-ended-FM_P-pipistrellus_Hte-Vienne_May1991_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'ppip_eval', exp_factor=10, template_type='evaluate')

#pnat: 7 pulses, file 066
bat_list1=(1,5,8,10,13,16,19)
bat_list2=(0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/066_QCF_flat-ended-FM_P-nathusii_Switzerland_April2008_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'pnat_eval', exp_factor=10, template_type='evaluate')

#pkuh: 3 pulses, file 071
bat_list1=(1,6,10)
bat_list2=(0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/071_flat-ended-FM_P-kuhlii_Hte-Vienne_May2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'pkuh_eval', exp_factor=10, template_type='evaluate')

#hsav_QCF: 7 pulses, file 074
bat_list1=(6,9,14,22,25,28,33)
bat_list2=(0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/074_QCF_capt_H-savii_Lozere_August1991_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'hsav_QCF_eval', exp_factor=10, template_type='evaluate')

#hsav: 5 pulses, file 075
bat_list1=(2,4,6,11,16)
bat_list2=(0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/075_flat-ended-FM_capt_H-savii_Alpes-Maritimes_July1993_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'hsav_eval', exp_factor=10, template_type='evaluate')

#hsav: 8 pulses, file 076
bat_list1=(20,27,31,36,45,48,53,56)
bat_list2=(0,0,0,0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/076_flat-ended-FM_H-savii_Alpes-Maritimes_July2009_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'hsav_eval', exp_factor=10, template_type='evaluate')

#mdas: 2 pulses, file 077
bat_list1=(1,3)
bat_list2=(0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/077_flat-ended-FM_M-dasycneme_Belgium_September1998_M-Van-de-Sijpe.wav',
                        bat_list1[i], bat_list2[i], 'mdas_eval', exp_factor=10, template_type='evaluate')

#eser_nasal: 5 pulses, file 077a
bat_list1=(3,6,11,14,17)
bat_list2=(0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/077a_FM-nasal_E-serotinus_understorey_Correze_August2012_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'eser_nasal_eval', exp_factor=10, template_type='evaluate')




#test files
name_file='080_FM-nasal_B-barbastellus_appr_FME-H1_Hte-Vienne_June2010_M-Barataud'

rectangles, regions, spectros=AD2.spect_loop('Barataud_CD_TimeExpansion10x/'+name_file + '.wav', exp_factor=10)

AD2.show_mregions(rectangles, spectros)



#bbar: 5 pulses, file 100
bat_list1=(0,3,4,6,9)
bat_list2=(0,0,0,0,0)
for i in range(len(bat_list1)):
    AD2.create_template('Barataud_CD_TimeExpansion10x/100_alt_capt_B-barbastellus_Charente-Mar_May2002_M-Barataud.wav',
                        bat_list1[i], bat_list2[i], 'bbar_eval', exp_factor=10, template_type='evaluate')




#evaluation plot
X_final, Y_final, net, D=AD4.evaluation_SOM(dim1=30, dim2=30, export='Evaluation_plot')

AD4.print_evaluate()
