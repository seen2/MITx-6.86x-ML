'''
while True:
        theta_prev=theta
        first_term=np.zeros(theta.shape)
        for i in range(X.shape[0]):
            xi=np.array([X[i]])
            ones=np.array([[1 if Y[i]==j else 0 for j in range(X.shape[0])]])
            result=compute_probabilities(xi,theta,temp_parameter)
            # ones=np.transpose(ones)
            # print(ones.shape,result.shape)
            first_term_truth=np.matmul(-result,ones)
            print(first_term_truth.shape,X.shape)
            first_term=first_term+np.matmul(first_term_truth,X)
        gradient=lambda_factor*theta-first_term/(temp_parameter*X.shape[0])
        theta=theta-alpha*gradient
        # print("***********************************************")
        # print("--------------FIRST_TERM--------------")
        # print(first_term)
        # print("--------------Prev--------------")
        # print(theta_prev)
        # print("---------------Theta----------------")
        # print(theta)
        # print("***********************************************")
        if np.allclose(theta,theta_prev):
            break

    return theta
'''