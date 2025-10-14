#Imports
import numpy as np
import pandas as pd

#Constants

w = float(6/(3+np.sqrt(7)))
#max number of loops
max_iter = 10
#b matrix

b = np.transpose((4,5,4))
#intial guess
x0 = np.zeros_like(b)
xtrue = np.array([1,1,1])
#A matrix
A = np.array([[3,1,0],
              [1,3,1],
              [0,1,3]])



def jacobi(A,b):
    x_old = x0.copy()
    x_new = np.zeros_like(x_old, dtype = float)
    D = np.diag(A)
    ND = 0
    error = []
    xjsave = []
    err_prev = 1
    for k in range(1,max_iter+1):
        for i in range(x_new.shape[0]):
            ND = 0
            for j in range (x_new.shape[0]):    
                if j != i:
                    ND += A[i][j] *x_old[j]
            x_new[i] = (b[i]-ND)/D[i]
        x_old = x_new.copy()
        err = x_old - xtrue

        errnorm = float(np.max(np.abs(err)))

        ratio = errnorm / err_prev 
        xjsave.append(x_old)
        error.append((k, errnorm, ratio))
        err_prev = errnorm

    return x_new, error, xjsave

def gauss_seidel(A, b):
    n = A.shape[0]
    x = np.zeros(n,dtype = float)
    error = []
    xgsave = []
    err_prev = 1

    for k in range(1, max_iter + 1):
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        xgsave.append(x.copy())
        err = x - xtrue
        
        errnorm = float(np.max(np.abs(err)))
        
        ratio = errnorm / err_prev 

        error.append((k, errnorm, ratio))
        err_prev = errnorm

    return x, error, xgsave

def SOR(A,b):
    n = A.shape[0]
    x_old = x0.copy()
    x_new = np.zeros(n, dtype = float)
    error = []
    xssave = []
    err_prev = 1
    for k in range(1,max_iter+1):
        for i in range(0,n):
            sumnew = 0
            sumold = 0
            for j in range(0,i):
                sumnew += A[i,j] * x_new[j]
            for j in range(i+1,n):
                sumold += A[i,j] * x_old[j]
            x_new[i] = (1-w)*x_old[i] +( w*( b[i] - (sumnew+sumold) ) / A[i,i])
        xssave.append(x_new.copy())
        err = xtrue - x_new

        errnorm= np.max(abs(err))
    
        ratio = errnorm/err_prev
    
        error.append((k, errnorm, ratio))
        err_prev = errnorm
        x_old = np.copy(x_new)
    return x_new, error, xssave


x_jacobi, hist_jacobi,x_jacobi_save = jacobi(A, b)
x_gs, hist_gs,x_gs_save = gauss_seidel(A, b)
x_sor, hist_sor,x_sor_save = SOR(A, b)

# Convert to tables
dfr_jacobi = pd.DataFrame(hist_jacobi, columns=["k","||e^(k)||∞", "ratio"])
dfx_jacobi =pd.DataFrame(x_jacobi_save, columns=["x1", "x2", "x3"])
dfr_gs = pd.DataFrame(hist_gs, columns=["k", "||e^(k)||∞", "ratio"])
dfx_gs = pd.DataFrame(x_gs_save, columns=["x1", "x2", "x3"])
dfr_sor = pd.DataFrame(hist_sor, columns=["k", "||e^(k)||∞", "ratio"])
dfx_sor = pd.DataFrame(x_sor_save, columns=["x1", "x2", "x3"])


print("\n=== Jacobi Method ===")
print(dfr_jacobi)
print("\n")
print(dfx_jacobi)
print(f"\nConverged in {len(hist_jacobi)} iterations.")

print(f"Final residual norm: {hist_jacobi[-1][1]:.6e}")

print("\n=== Gauss-Seidel Method ===")
print(dfr_gs)
print("\n")
print(dfx_gs)
print(f"\nConverged in {len(hist_gs)} iterations.")
print(f"Final residual norm: {hist_gs[-1][1]:.6e}")
print("\n=== SOR Method ===")
print(dfr_sor)
print("\n")
print(dfx_sor)
print(f"\nConverged in {len(hist_gs)} iterations.")
print(f"Final residual norm: {hist_gs[-1][1]:.6e}")

