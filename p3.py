import numpy as np
from scipy import linalg


def LU_solve(A,b):#The function takes a matrix A and a vector b and returns the vector x, the solution to Ax=b
    r = 0 #This variable holds the current row that is being considered, goes from 0 to n-1
    c = 0
    size = min(A.shape) #the number of rows in the matrix
    rows = A.shape[0]
    cols = A.shape[1]
    while c<(cols): #iterate through all rows
        if A[r][c]!=0: #check if the pivot is non-zero
            for i in range(r+1,rows): #if it is non zero create L matrices for the column
                if A[i][r]!=0:
                    L_c = L_mat(r,i,rows,A[i][c]/A[r][c]) #this function calculates the L matrix for A[i][r]
                    A = np.dot(L_c,A) #apply the row operation to A
                    b = np.dot(L_c,b) #apply the row operation to b
            r+=1 #move on to next row
            c+=1
            print (c)
        else: #if there is no pivot need to permute rows
            ind = first_nonzero(A[:,c],r,rows) #this function returns the index of the first non-zero entry below the pivot in that column
            print (ind)
            if ind == -1: #if there is no non-zero entry then the matrix is singular
                c+=1
            else:
                P = perm_mat(ind,r,rows)# this function calculates the permutation matrix to swap row r with the first row below with non-zero pivot
                A = np.dot(P,A) #apply the permutation the A
                b = np.dot(P,b) #apply the permutation the b
    #if abs(A[size-1][size-1])<10.0**(-10):#the final pivot of a singular matrix can often be non-zero due to rounding errors so if the final pivot is small enough claim that the matrix is singular
        #return ("unavoidable zero pivot")
    #x= back_subs(A,b,size)# solve Ux=c by back substitution
    #return x
    return A


def L_mat(r,i,rows,factor): #this creates the L matrix for the gi
    L = np.eye(rows)
    L[i][r] =  -factor
    return L

def first_nonzero(col,r,rows): #finds the first non-zero entry below the pivot
    for i in range(r,rows):
        if col[i]!=0:
            return i
    return -1 #if there is no non-zero entry return -1

def perm_mat(i,r,rows): #this creates the permuation matrix to swap rows i and j
    P = np.eye(rows)
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

A = np.array([[1.,6.,-3.,0.],[0.,4.,2.,-3.],[3.,18.,1.,-5.],[2.,0.,0.,3.],[2.,8.,2.,0.]])
b = np.array([1,2,4,7,7])
print (LU_solve(A,b))
