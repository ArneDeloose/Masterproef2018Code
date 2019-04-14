
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD1_Loading as AD1
import AD2_Spectro as AD2

path='C:/Users/arne/Documents/School/Thesis'; #Change this to directory that stores the data
os.chdir(path)


freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp=AD1.loading_init()

name_file='001_FMa-CF-FMd_R-ferrumequinum_Lozere_August1991_M-Barataud'

rectangles, regions, spectros=AD2.spect_loop('Barataud_CD_TimeExpansion10x/'+name_file + '.wav', exp_factor=10)

AD2.show_mregions(rectangles, spectros)
