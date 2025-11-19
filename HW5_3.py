import numpy as np
import matplotlib.pyplot as plt

max_iter = 100000
tol = 1e-7
p_min = 1
p_max = 14
a = 0
b = 1.0
phi_a = 1.0
phi_b = np.e

# From previous HW
def GD(A, b, x0):
    x = x0.copy().astype(float)
    for k in range(1, max_iter + 1):
        r = b - A @ x
        rr = float(r @ r)
        if rr < tol**2:
            return x, k, np.sqrt(rr)
        Ar = A @ r
        denom = float(r @ Ar)

        alpha = rr / denom
        x += alpha * r
    r = b - A @ x
    return x, max_iter, np.linalg.norm(r)

# From previous HW
def CG(A, b, x0):
    x = x0.copy().astype(float)
    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)

    if rs_old < tol**2:
        return x, 0, np.sqrt(rs_old)

    for k in range(1, max_iter + 1):
        Ap = A @ p
        pAp = float(p @ Ap)
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap

        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            return x, k, np.sqrt(rs_new)

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, max_iter, np.sqrt(rs_old)

# Modified from previous HW to handle general cases
def LU_tridiag(a, b, c, d):
    n = len(b)
    ac = a.astype(float).copy()
    bc = b.astype(float).copy()
    cc = c.astype(float).copy()
    dc = d.astype(float).copy()

    for i in range(1, n):
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    x = np.zeros(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x

# Build system for solving the BVP
def build_system(q_func, N):

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    xi = x[1:-1]  # interior points
    q_i = q_func(xi)

    n = N - 1
    # tridiagonal entries for interior system
    a_diag = -1.0 * np.ones(n)          
    b_diag = (2.0 + (h ** 2) * q_i)    
    c_diag = -1.0 * np.ones(n)          

    # RHS collects boundary contributions
    rhs = np.zeros(n, dtype=float)
    rhs[0] += phi_a
    rhs[-1] += phi_b

    # Build explicit dense A for iterative solvers
    A = np.zeros((n, n), dtype=float)
    # main diagonal
    A[np.arange(n), np.arange(n)] = b_diag

    if n > 1:
        A[np.arange(1, n), np.arange(n - 1)] = a_diag[1:]
        A[np.arange(n - 1), np.arange(1, n)] = c_diag[:-1]

    return x, a_diag, b_diag, c_diag, rhs, A

def q(x):
    return 4.0 * x ** 2 + 2.0

def exact_phi(x):
    return np.exp(x ** 2)
x_ana = np.linspace(0,1,num = 50)


print("Tolerance =",tol)
print("Max Iterations =",max_iter)

header = ("Method |      h        ||u-phi||∞      /h             /h^2             /h^3")
print(header)
print("-"*len(header))

LU_plots = []
GD_plots = []
CG_plots = []
x_plots = []
h_vals = []

for p in range(p_min, p_max + 1):
    h = 2.0 ** (-p)
    N = int(round((b - a) / h))

    # build system
    x_grid, a_diag, b_diag, c_diag, rhs, A = build_system(q, N)
    n = N - 1

    # Solve with LU (Thomas)
    U_interior_LU = LU_tridiag(a_diag, b_diag, c_diag, rhs)

    # RHS for iterative solvers
    b_vec = rhs.copy().astype(float)
    x0 = np.zeros(n, dtype=float)

    # GD
    U_int_GD, it_gd, res_gd = GD(A, b_vec, x0)

    # CG
    U_int_CG, it_cg, res_cg = CG(A, b_vec, x0)

    # Build full solutions
    U_LU = np.zeros(N + 1); U_LU[0]=phi_a; U_LU[-1]=phi_b; U_LU[1:-1]=U_interior_LU
    U_GD = np.zeros(N + 1); U_GD[0]=phi_a; U_GD[-1]=phi_b; U_GD[1:-1]=U_int_GD
    U_CG = np.zeros(N + 1); U_CG[0]=phi_a; U_CG[-1]=phi_b; U_CG[1:-1]=U_int_CG
    x_plots.append(x_grid)
    h_vals.append(h)
    LU_plots.append(U_LU)
    GD_plots.append(U_GD)
    CG_plots.append(U_CG)

    exact_vals = exact_phi(x_grid)

    # Inf norms
    err_LU = np.max(np.abs(U_LU - exact_vals))
    err_GD = np.max(np.abs(U_GD - exact_vals))
    err_CG = np.max(np.abs(U_CG - exact_vals))

    # Print
    print("P =",p)
    print(f"  LU   | {h:10.2e}  {err_LU:12.3e}  {err_LU/h:12.3e}  {err_LU/h**2:12.3e}  {err_LU/h**3:12.3e}")
    print(f"  GD   | {h:10.2e}  {err_GD:12.3e}  {err_GD/h:12.3e}  {err_GD/h**2:12.3e}  {err_GD/h**3:12.3e}")
    print(f"  CG   | {h:10.2e}  {err_CG:12.3e}  {err_CG/h:12.3e}  {err_CG/h**2:12.3e}  {err_CG/h**3:12.3e}")
    print("-"*len(header))

# ---- LU plot ----
plt.figure(figsize=(10,6))
for i, U in enumerate(LU_plots):
    plt.plot(x_plots[i], U, label=f"p={p_min+i}")
plt.plot(x_ana,exact_phi(x_ana), label = "True")
plt.title("LU Solutions u_h(x) for different h")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.show()

# ---- GD plot ----
plt.figure(figsize=(10,6))
for i, U in enumerate(GD_plots):
    plt.plot(x_plots[i], U, label=f"p={p_min+i}")
plt.plot(x_ana,exact_phi(x_ana), label = "True")
plt.title("GD Solutions u_h(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.show()

# ---- CG plot ----
plt.figure(figsize=(10,6))
for i, U in enumerate(CG_plots):
    plt.plot(x_plots[i], U, label=f"p={p_min+i}")
plt.plot(x_ana,exact_phi(x_ana), label = "True")
plt.title("CG Solutions u_h(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.show()

