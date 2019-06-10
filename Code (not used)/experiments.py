
#make sure there are folders for every type in the experiment templates
#(empty folders aren't copied from Github, but are needed)

#initial setting
import os
path='C:/Users/arne/Documents/Github/Masterproef2018Code/Data/Modules';
os.chdir(path)
import AD4_SOM as AD4
import AD5_MDS as AD5
import matplotlib.pyplot as plt
import numpy as np

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


kappa_matrix1=np.zeros((9,3))
kappa_matrix2=np.zeros((9,3))

#MDS
X_transform=np.matmul(D, X_final)
dist=AD5.calc_dist_matrix(X_transform, 1)
pos=AD5.calc_pos(dist)
s = 10
plot1=plt.scatter(pos[0:6, 0], pos[0:6, 1], color='turquoise', marker='o', s=s, lw=1, label='1')
plot2=plt.scatter(pos[6:11, 0], pos[6:11, 1], color='red', marker='>', s=s, lw=1, label='2')
plot3=plt.scatter(pos[11:23, 0], pos[11:23, 1], color='green', marker='v', s=s, lw=1, label='3')
plt.legend(handles=[plot1, plot2, plot3])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.close()

plot4=plt.scatter(pos[23:29, 0], pos[23:29, 1], color='turquoise', marker='o', s=s, lw=1, label='1')
plot5=plt.scatter(pos[29:34, 0], pos[29:34, 1], color='red', marker='>', s=s, lw=1, label='2')
plot6=plt.scatter(pos[34:, 0], pos[34:, 1], color='green', marker='v', s=s, lw=1, label='3')
plt.legend(handles=[plot4, plot5, plot6])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.close()


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
    

ax=plt.subplot(1,1,1)
p1,=ax.plot(range(1,10), kappa_matrix1[:, 0],  '-ko', label='eser')
p2,=ax.plot(range(1,10), kappa_matrix1[:, 1], '-k>', label='nlei')
p3,=ax.plot(range(1,10), kappa_matrix1[:, 2], '-k*', label='ppip')

p1,=ax.plot(range(1,10), kappa_matrix2[:, 0], '-ro', label='eser (val)')
p2,=ax.plot(range(1,10), kappa_matrix2[:, 1], '-r>', label='nlei (val)')
p3,=ax.plot(range(1,10), kappa_matrix2[:, 2], '-r*', label='ppip (val)')

plt.xlabel('Number of neighbors (K)')
plt.ylabel('Cohen\'s kappa')

legend_x = 1
legend_y = 0.5
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))

plt.savefig('KNN_plot.eps', format='eps', dpi=1000)
plt.show()
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels)
plt.close()


#experiment 2.2: DML
path1='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_1'
path2='C:/Users/arne/Documents/Github/Masterproef2018Code/Experiment_templates/Experiment1_2'

#fit som and dml, reg and dml data
X_final, Y_final, net, D=AD4.evaluation_SOM(path=path1, dim1=5, dim2=5, Plot_Flag=False)

#eval Data, insert previous som and dml
X_final_eval, Y_final_eval, net, D=AD4.evaluation_SOM(path=path2, dim1=5, dim2=5, Plot_Flag=False,
                                                      SOM=net, dml=D, templates_features=path1)

D2=np.identity(30)

#calculate kappa1 (DML)
PA, match_scores=AD4.KNN_calc(X_final_eval, Y_final_eval, D, path=path2, K=5)
PE=AD4.calc_PE(path=path2)
kappa1=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa1, path=path2)

for i in range(7): #normalise data
    X_final_eval[i,:]=(X_final_eval[i,:]-np.min(X_final_eval[i,:]))/(np.max(X_final_eval[i,:])-np.min(X_final_eval[i,:]))


#calculate kappa2 (Euclidean)
PA, match_scores=AD4.KNN_calc(X_final_eval, Y_final_eval, D2, path=path2, K=5)
PE=AD4.calc_PE(path=path2)
kappa2=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa2, path=path2)

#cluster seperately
#freq
D3=D[0:7,0:7]
PA, match_scores=AD4.KNN_calc(X_final_eval[0:7, :], Y_final_eval, D3, path=path2, K=5)
PE=AD4.calc_PE(path=path2)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path2)

#ssim
D4=D[7:,7:]
PA, match_scores=AD4.KNN_calc(X_final_eval[7:, :], Y_final_eval, D4, path=path2, K=5)
PE=AD4.calc_PE(path=path2)
kappa=AD4.calc_kappa(PA, PE)
AD4.print_evaluate2(PA, kappa, path=path2)


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




