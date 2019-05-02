import numpy as np

def KNN_calc(X_final, Y_final, D, m):
    dist_mat=np.zeros((X_final.shape[1],)) #distances matrix, temporarily
    matches=np.zeros((X_final.shape[1],)) #saves the matches, temporarily
    match_scores=np.zeros((X_final.shape[1],)) #final score individual
    final_scores=np.zeros((np.max(Y_final),)) #final score type
    
    #temp variables
    count_i=0
    temp_1=0
    temp_2=0
    for i in range(X_final.shape[1]): #go through all datapoints
        for j in range(X_final.shape[1]): #check with each other datapoint
            dist_mat[j]=np.dot((X_final[i,:]-X_final[j,:])**2)
            for k in range(m):
                #sort ascending, take k lowest values
                #start at 1 because 0 is the point i (matches with itself, distance 0)
                index_match=[dist_mat.index(x) for x in sorted(dist_mat)[k+1]]
                matches[k]=Y_final[index_match]
            #calculate score for this point
            count_i=0
            for l in range(m):
                count_i+=1
            match_scores[i]=count_i/m
    #convert scores to take average per type
    for n1 in range(np.max(Y_final)):
        temp_1=0
        temp_2=0
        for n2 in range(X_final.shape[1]): #go through all datapoints
            if Y_final[n2]==n1: #match 
                temp_1+=1 #number of matchers
                temp_2+=match_scores[n2] #score
        final_scores[n1]=temp_2/temp_1 #average score
    return(final_scores, match_scores)



