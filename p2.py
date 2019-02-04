import numpy as np
from scipy import linalg

def LU_solve(A,b):#The function takes a matrix A and a vector b and returns the vector x, the solution to Ax=b
    r = 0 #This variable holds the current row that is being considered, goes from 0 to n-1
    size = len(A) #the number of rows in the matrix
    while r<(size-1): #iterate through all rows
        if A[r][r]!=0: #check if the pivot is non-zero
            for i in range(r+1,size): #if it is non zero create L matrices for the column
                if A[i][r]!=0:
                    L_c = L_mat(r,i,size,A[i][r]/A[r][r]) #this function calculates the L matrix for A[i][r]
                    A = np.dot(L_c,A) #apply the row operation to A
                    b = np.dot(L_c,b) #apply the row operation to b
            r+=1 #move on to next row
        else: #if there is no pivot need to permute rows
            ind = first_nonzero(A[:,r],r,size) #this function returns the index of the first non-zero entry below the pivot in that column
            if ind == -1: #if there is no non-zero entry then the matrix is singular
                return ("unavoidable zero pivot")
            P = perm_mat(ind,r,size)# this function calculates the permutation matrix to swap row r with the first row below with non-zero pivot

            A = np.dot(P,A) #apply the permutation the A
            b = np.dot(P,b) #apply the permutation the b
    if abs(A[size-1][size-1])<10.0**(-10):#the final pivot of a singular matrix can often be non-zero due to rounding errors so if the final pivot is small enough claim that the matrix is singular
        return ("unavoidable zero pivot")
    x = back_subs(A,b,size)# solve Ux=c by back substitution
    return x


def L_mat(r,i,size,factor): #this creates the L matrix for the gi
    L = np.eye(size)
    L[i][r] =  -factor
    return L

def first_nonzero(col,r,size): #finds the first non-zero entry below the pivot
    for i in range(r,size):
        if col[i]!=0:
            return i
    return -1 #if there is no non-zero entry return -1

def perm_mat(i,r,size): #this creates the permuation matrix to swap rows i and j
    P = np.eye(size)
    P[i][i] = 0
    P[r][r] = 0
    P[r][i] = 1
    P[i][r] = 1
    return P

def forward_subs(L,b,size): #solves Lx=b for x by forward substitution
    x = np.zeros(size)
    x[0] = b[0]/L[0][0]
    for i in range(1,size):
        sum = 0
        for j in range(i):
            sum += L[i][j]*x[j]
        x[i] = (b[i]-sum)/L[i][i]
    return x


def back_subs(U,b,size): #solves Ux=b for x by backward substitution
    x = np.zeros(size)
    x[size-1] = b[size-1]/U[size-1][size-1]
    for i in range(1,size):
        sum = 0
        for j in range(i):
            sum += U[size-1-i][size-1-j]*x[size-1-j]
        x[size-1-i] = (b[size-1-i]-sum)/U[size-1-i][size-1-i]
    return x



A3 = np.array([[3.,6.,-3.,2.],[2.,5.,0.,4.],[3.,9.,3.,-1.],[1.,2.,-1.,1.]])
b3 = np.array([3.,1.,-3.,2.])


if np.linalg.matrix_rank(A3) == len(A3):
    print (LU_solve(A3,b3))
else:
    combined = np.zeros( (len(A3),len(A3)+1))
    combined[:,:-1] = A3
    combined[:,-1] = b3
    if np.linalg.matrix_rank(combined) == np.linalg.matrix_rank(A3):
        print ("Infinitely many solutions")
    else:
        print ("No solutions")
