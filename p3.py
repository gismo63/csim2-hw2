import numpy as np
from scipy import linalg


def null_col(A):#The function takes a matrix A and a vector b and returns the vector x, the solution to Ax=b
    A_init = A
    r = 0 #This variable holds the current row that is being considered, goes from 0 to n-1
    c = 0
    rank = 0
    pivots = []
    size = min(A.shape) #the number of rows in the matrix
    rows = A.shape[0]
    cols = A.shape[1]
    while c<(cols): #iterate through all rows
        if A[r][c]!=0: #check if the pivot is non-zero
            for i in range(r+1,rows): #if it is non zero create L matrices for the column
                if A[i][r]!=0:
                    L_c = L_mat(r,i,rows,A[i][c]/A[r][c]) #this function calculates the L matrix for A[i][r]
                    A = np.dot(L_c,A) #apply the row operation to A
            pivots.append(c)
            r+=1 #move on to next row
            c+=1
        else: #if there is no pivot need to permute rows
            ind = first_nonzero(A[:,c],r,rows) #this function returns the index of the first non-zero entry below the pivot in that column
            if ind == -1: #if there is no non-zero entry then the matrix is singular
                c+=1
            else:
                P = perm_mat(ind,r,rows)# this function calculates the permutation matrix to swap row r with the first row below with non-zero pivot
                A = np.dot(P,A) #apply the permutation the A
    out = []
    if len(pivots) == cols:
        out.append(np.zeros(cols))
    else:
        null_basis = np.zeros((cols,cols - len(pivots)))
        for i in range(cols - len(pivots)):
            null_basis[:,i] = back_subs_null(A,cols,rows,pivots,i)
        out.append(null_basis)
    col_space = np.zeros((rows,len(pivots)))
    for i in range(len(pivots)):
        col_space[:,i] = A_init[:,pivots[i]]
    out.append(col_space)
    return out


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


def back_subs_null(U,cols,rows,pivots,count): #solves Ux=b for x by backward substitution
    num_free = 0
    piv = 1
    x = np.zeros(cols)
    for i in range(cols):
        if pivots[-piv] == cols-i-1:
            sum = 0
            for j in range(i):
                sum += U[rows-1-i+num_free,cols-1-j]*x[cols-1-j]
            x[cols-i-1] = -float(sum)/U[rows-1-i+num_free,cols-1-i]
            piv+=1
        elif count == num_free:
            x[cols-i-1] = 1
            num_free+=1
        else:
            x[cols-i-1] == 0
            num_free+=1
    return x

A = np.array([[1.,6.,-3.,0.],[0.,4.,2.,-3.],[3.,18.,1.,-5.],[2.,0.,0.,3.],[2.,8.,2.,0.]])
print ("Null Space: ")
null,col = null_col(A)
print (null)
print ("Column Space: ")
for i in range(col.shape[1]):
    print (col[:,i])

print ("Left Null Space: ")
null,col = null_col(np.transpose(A))
print (null)
print ("Row Space: ")
for i in range(col.shape[1]):
    print (col[:,i])
