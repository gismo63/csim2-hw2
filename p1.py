import numpy as np
from scipy import linalg

def pseud_inv(A):
    dims = np.array(A1.shape)
    rank = np.linalg.matrix_rank(A)
    if rank<min(dims):
        print ("No Inverse/Pseudoinverse Exists\n\n")
    else:
        p_inv = linalg.pinv(A)
        print ("Pseudoinverse =")
        print (p_inv)
        if dims[0]<dims[1]:
            print ("A*A+ =")
            print (np.dot(A,p_inv).round(10))
        else:
            print ("A*A+ =")
            print (np.dot(p_inv,A).round(10))


A1 = np.array([[6.,1.,5.],[5.,1.,4.],[0.,5.,-5.],[2.,2.,0.]])

A2 = np.array([[6.,1.,5.],[5.,1.,4.],[1.,5.,-5.],[2.,2.,0.]])

pseud_inv(A1)
pseud_inv(A2)
