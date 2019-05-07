
#initial setting
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD4_SOM as AD4

path0='C:/Users/arne/Documents/GitHub/Masterproef2018Code/Data'; #Change this to directory that stores the data
os.chdir(path0)


#Experiment 1
path1='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_1'
path2='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_2'

#fit som and dml, reg and dml data
X_final, Y_final, net, D=AD4.evaluation_SOM(path=path1, dim1=5, dim2=5, Plot_Flag=False)

#eval Data, insert previous som and dml
X_final_eval, Y_final_eval, net, D=AD4.evaluation_SOM(path=path2, dim1=5, dim2=5, Plot_Flag=False,
                                                      SOM=net, dml=D, templates_features=path1)

#calculate kappa1
PA, match_scores=AD4.KNN_calc(X_final, Y_final, D, path=path1)
PE=AD4.calc_PE(path=path1)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path1)

#calculate kappa2
PA, match_scores=AD4.KNN_calc(X_final_eval, Y_final_eval, D, path=path2)
PE=AD4.calc_PE(path=path2)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path2)



#Experiment 2



#experiment 3

#identity matrix D


