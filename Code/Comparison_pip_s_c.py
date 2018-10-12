

s_mat1=AD.create_smatrix(rectangles1, spectros1, 1)
s_mat1=AD.calc_smatrix(s_mat1, regions1, templates_0, 0)

c_mat1=AD.create_cmatrix(rectangles1, spectros1)
c_mat1=AD.calc_cmatrix(c_mat1, s_mat1)
res1=AD.calc_result(c_mat1, 1)

#ppip test data
s_mat2=AD.create_smatrix(rectangles2, spectros2, 1)
s_mat2=AD.calc_smatrix(s_mat2, regions2, templates_0, 0)

c_mat2=AD.create_cmatrix(rectangles2, spectros2)
c_mat2=AD.calc_cmatrix(c_mat2, s_mat2)
res2=AD.calc_result(c_mat2, 1)


#esper bat test data
s_mat3=AD.create_smatrix(rectangles3, spectros3, 1)
s_mat3=AD.calc_smatrix(s_mat3, regions3, templates_0, 0)

c_mat3=AD.create_cmatrix(rectangles3, spectros3)
c_mat3=AD.calc_cmatrix(c_mat3, s_mat3)
res3=AD.calc_result(c_mat3, 1)
