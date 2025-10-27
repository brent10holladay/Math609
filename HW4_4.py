#Imports
import numpy as np

#Constants
max_iter=100000
tol = 1e-8
## PART A ##
print("\nPART A\n")
# Make the A and b matrix
def generate_system(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (1 + (i+1) + (j+1))
    b = np.zeros(n)
    for i in range(n):
        b[i] = (1/3) * np.sum(A[i, :])
    return A, b

# Gradient Descent method
def GD(A, b, x0,alpha):
    x = x0.copy()
    for k in range(max_iter):
        r = A @ x - b
        grad_norm = np.linalg.norm(r)
        if grad_norm < tol:
            break
        x -= alpha * r
    return x, k, grad_norm

# Conjugate Gradient method
def CG(A, b, x0):

    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, k, np.sqrt(rs_new)

# Use methods for 16x16 and 32x32 matrix
for n in [16, 32]:
    print(f"\n{n}x{n} Matrix\n")
    A, b = generate_system(n)
    x0 = np.zeros(n)
    if n == 16:
        alpha = 1.62222
    if n == 32:
        alpha = 1.38393
    x_gd, it_gd, err_gd = GD(A, b, x0,alpha)
    x_cg, it_cg, err_cg = CG(A, b, x0)
    
    print("Optimal alpha:",np.real(2/(max(np.linalg.eig(A)[0])+min(np.linalg.eig(A)[0]))))
    print(f"=== Gradient Descent ===\nx = {x_gd}\niterations = {it_gd}, final residual norm = {err_gd:.2e}\n")
    print(f"=== Conjugate Gradient ===\nx = {x_cg}\niterations = {it_cg}, final residual norm = {err_cg:.2e}")


## PART B ##
print("\n\nPART B\n\n")
print("Tolerance = 1e-8\nMax iterations = 1000")
# Make the A and b matrix

A = np.array([
    [10,  1,  2,  3,  4],
    [ 1,  9, -1,  2, -3],
    [ 2, -1,  7,  3, -5],
    [ 3,  2,  3, 12, -1],
    [ 4, -3, -5, -1, 15]
    ], dtype=float)

# Define vector b
b = np.array([12, -27, 14, -17, 12], dtype=float)

# Add jacobi and GS method
max_iter=1000
tol = 1e-10
alpha = 0.09461
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
    return x_new, k, residuals

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

    return x, k,residuals

x_jacobi,it_jacobi, hist_jacobi = jacobi(A, b)
x_gs, it_gs, hist_gs = gauss_seidel(A, b)
x_gd, it_gd, err_gd = GD(A, b,x0, alpha)
x_cg, it_cg, err_cg = CG(A, b,x0)




print("\n=== Jacobi Method ===")
print(x_jacobi)
print(f"\nConverged in {it_jacobi} iterations.")
print(f"Final residual norm: {hist_jacobi[-1][1]:.6e}")

print("\n=== Gauss-Seidel Method ===")
print(x_gs)
print(f"\nConverged in {it_gs} iterations.")
print(f"Final residual norm: {hist_gs[-1][1]:.6e}")

print("\n=== Gradient Descent Method ===")
print("Optimal alpha:",np.real(2/(max(np.linalg.eig(A)[0])+min(np.linalg.eig(A)[0]))))
print(x_gd)
print(f"\nConverged in {it_gd} iterations.")
print(f"Final residual norm: {err_gd:.6e}")

print("\n=== Conjugate Gradient Method ===")
print(x_cg)
print(f"\nConverged in {it_cg} iterations.")
print(f"Final residual norm: {err_cg:.6e}")
