
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
PA, match_scores=AD4.KNN_calc(X_final, Y_final, D, path=path1, K=5)
PE=AD4.calc_PE(path=path1)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path1)

#calculate kappa2
PA, match_scores=AD4.KNN_calc(X_final_eval, Y_final_eval, D, path=path2, K=5)
PE=AD4.calc_PE(path=path2)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path2)



#Experiment 2.1: K
path1='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_1'
path2='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_2'

#fit som and dml, reg and dml data
X_final, Y_final, net, D=AD4.evaluation_SOM(path=path1, dim1=5, dim2=5, Plot_Flag=False)

#eval Data, insert previous som and dml
X_final_eval, Y_final_eval, net, D=AD4.evaluation_SOM(path=path2, dim1=5, dim2=5, Plot_Flag=False,
                                                      SOM=net, dml=D, templates_features=path1)

import numpy as np

kappa_matrix1=np.zeros((9,3))
kappa_matrix2=np.zeros((9,3))


#calculate kappa1
for i in range(1, 10):
    PA, match_scores=AD4.KNN_calc(X_final, Y_final, D, path=path1, K=i)
    PE=AD4.calc_PE(path=path1)
    kappa_matrix1[i-1, :]=AD4.calc_kappa(PA, PE)

#calculate kappa2
for i in range(1, 10):
    PA, match_scores=AD4.KNN_calc(X_final_eval, Y_final_eval, D, path=path2, K=i)
    PE=AD4.calc_PE(path=path2)
    kappa_matrix2[i-1, :]=AD4.calc_kappa(PA, PE)
    
import matplotlib.pyplot as plt

ax=plt.subplot(1,1,1)
p1,=ax.plot(range(1,10), kappa_matrix1[:, 0],  '-ro', label='eser')
p2,=ax.plot(range(1,10), kappa_matrix1[:, 1], '-g>', label='nlei')
p3,=ax.plot(range(1,10), kappa_matrix1[:, 2], '-b*', label='ppip')

p1,=ax.plot(range(1,10), kappa_matrix2[:, 0], '-cv', label='eser (eval)')
p2,=ax.plot(range(1,10), kappa_matrix2[:, 1], '-m^', label='nlei (eval)')
p3,=ax.plot(range(1,10), kappa_matrix2[:, 2], '-kd', label='ppip (eval)')

plt.xlabel('Number of neighbors (K)')
plt.ylabel('Cohen\'s kappa')
plt.legend()
plt.savefig('KNN_plot.eps', format='eps', dpi=1000)
plt.show()
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels)
plt.close()


#experiment 2.2: DML

#identity matrix D


#experiment 3

path1='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_1'
path2='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment3'

#fit som and dml, reg and dml data
X_final, Y_final, net, D=AD4.evaluation_SOM(path=path1, dim1=5, dim2=5, Plot_Flag=False)

#eval Data, insert previous som and dml
X_final_eval, Y_final_eval, net, D=AD4.evaluation_SOM(path=path2, dim1=5, dim2=5, Plot_Flag=False,
                                                      SOM=net, dml=D, templates_features=path1)

#calculate kappa1
PA, match_scores=AD4.KNN_calc(X_final, Y_final, D, path=path1, K=5)
PE=AD4.calc_PE(path=path1)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path1)

#calculate kappa2
PA, match_scores=AD4.KNN_calc(X_final_eval, Y_final_eval, D, path=path2, K=5)
PE=AD4.calc_PE(path=path2)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path2)




