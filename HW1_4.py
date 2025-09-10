##Imports##
import numpy as np
import matplotlib.pyplot as plt

##Constants##
e = 10**(-3)
plt.figure(figsize=(10, 6))


##Algorithm (Thomas)##

#For n = 1...8
for n in range(1,9):
    #Define h
    h = 2**(-n)
    #Define N
    N = (2**n) - 1
    #Define f
    f = np.zeros([N,1])
    for i in range(0,N):
        f[i] = 2*i*h + 1
    #Define middle matrix for i
    diag= (2*e + (h**2))/(h**2)
    #Define
    side = -e/(h**2)
    y = np.zeros([N,1])
    u = np.zeros([N,1])
    #initial
    y[0] = f[0]
    u[0] = diag
    for k in range(1,N):
        lk = side/u[k-1]
        y[k] = f[k] - lk*y[k-1]
        u[k] = diag - lk*side
    x = np.zeros([N,1])
    x[-1] =  y[-1]/u[-1]
    for k in range(N-2,0,-1):
        x[k] = (y[k] - side*x[k+1])/u[k]
    
    plt.plot(np.linspace(1,x.shape[0],x.shape[0])*h ,x, marker='o',label = n)
    plt.xlabel('x_i, where x_i = ih')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)

    plt.title('Solution for n = 1,2,...,8')
plt.show()
