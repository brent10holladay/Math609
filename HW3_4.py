#Imports
import numpy as np
import pandas as pd

#Constants

#Identity matrix

I = np.identity(3)
#B matrix
B = np.array([[4,-1,0],
              [-1,4,-1],
              [0 ,-1, 4]], dtype= float)
b = np.transpose((0,0,1,0,0,1,0,0,1))
#0 matrix
z = np.zeros((3,3))
#A matrix
A = np.block([[B, -I, z],
             [-I,B,-I],
             [z,-I,B]])
#max number of loops
max_iter = 100
#termination criteria
tol = 1e-2
#intial guess
x0 = np.zeros_like(b)

def jacobi(A,b):
    x_old = x0.copy()
    x_new = np.zeros_like(x_old, dtype = float)
    D = np.diag(A)
    ND = 0
    residuals = []
    r_prev = 1
    for k in range(1,max_iter+1):
        for i in range(x_new.shape[0]):
            ND = 0
            for j in range (x_new.shape[0]):    
                if j != i:
                    ND += A[i][j] *x_old[j]
            x_new[i] = (b[i]-ND)/D[i]
        x_old = x_new.copy()
        r = b - A @ x_new
        rnorm = float(np.max(np.abs(r)))
        
        ratio = rnorm / r_prev 

        residuals.append((k, rnorm, ratio))
        r_prev = rnorm
        if rnorm < tol:
            break
    return x_new, residuals

def gauss_seidel(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    residuals = []
    r_prev = 1

    for k in range(1, max_iter + 1):
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        r = b - A @ x
        rnorm =  float(np.max(np.abs(r)))



        ratio = rnorm / r_prev 

        residuals.append((k, rnorm, ratio))

        if rnorm < tol:
            break

        r_prev = rnorm

    return x, residuals
x_jacobi, hist_jacobi = jacobi(A, b)
x_gs, hist_gs = gauss_seidel(A, b)

# Convert to tables
df_jacobi = pd.DataFrame(hist_jacobi, columns=["k","||r^(k)||∞", "ratio"])
df_gs = pd.DataFrame(hist_gs, columns=["k", "||r^(k)||∞", "ratio"])


print("\n=== Jacobi Method ===")
print(df_jacobi)
print(f"\nConverged in {len(hist_jacobi)} iterations.")
print(f"Final residual norm: {hist_jacobi[-1][1]:.6e}")

print("\n=== Gauss-Seidel Method ===")
print(df_gs)
print(f"\nConverged in {len(hist_gs)} iterations.")
print(f"Final residual norm: {hist_gs[-1][1]:.6e}")
