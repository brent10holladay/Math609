##Imports##
import numpy as np
import matplotlib.pyplot as plt

##Make the first matrix##
A_1 = np.zeros([4,4])
for i in range(0,4):
    for j in range(0,4):
        if i == j:
            A_1[i,j] = 2
        if (j == i-1) or (j == i+1):
            A_1[i,j] = -1
print("First Matrix = \n",A_1)
        
##Factorize first matrix##

#Start with empty lower and upper matrices
N = A_1.shape[0]
lower = np.zeros(A_1.shape)
upper = np.zeros(A_1.shape)

#Make def to use for matrix 2
def CholFact(A):
    for i in range(0,N):
        for j in range(0,i+1):
            
            tot = 0
            for k in range(0,j):
                tot += lower[i,k]*lower[j,k]
    
            if i==j:
                lower[i,j] = np.sqrt(A[i,i] - tot)
            else:
                lower[i,j] = (1/lower[j,j])*(A[i,j] - tot)
            upper = lower.transpose()
    print("\nLower = \n",lower)
    print("\nUpper = \n",upper)
    
#Print results
CholFact(A_1)

##Make the second matrix##
A_2 = np.zeros([4,4])
for i in range(0,4):
    for j in range(0,4):
        A_2[i,j] = 1/(i+j+1)
print("\n\nSecond Matrix =\n ",A_2)

##Factorize the second matrix##

#Use the factorization def
N = A_2.shape[0]
lower = np.zeros(A_2.shape)
upper = np.zeros(A_2.shape)
#Print results
CholFact(A_2)
