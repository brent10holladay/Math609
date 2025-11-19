import numpy as np
import matplotlib.pyplot as plt


def LU_tridiag(a, b, c, d):
    n = len(b)
    ac = a.copy().astype(float)
    bc = b.copy().astype(float)
    cc = c.copy().astype(float)
    dc = d.copy().astype(float)

    for i in range(1, n):
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x

def q(x):
    return 4*x**2 + 2

def exact_phi(x):
    return np.exp(x**2)

p_min= 1
p_max = 6

a, b = 0.0, 1.0
phi_a = 1.0
phi_b = np.e

header =("    h       ||u-phi||∞       /h^3          /h^4          /h^5")
print(header)
print("-"*len(header))

LU_plots = []
x_plots = []

for p in range(p_min, p_max+1):
    h = 2.0**(-p)
    N = int((b - a)/h)
    x = np.linspace(a, b, N+1)

    q_im1 = q(x[:-2])      
    q_i   = q(x[1:-1])     
    q_ip1 = q(x[2:])       
    n = N - 1

    a_coef = -1.0 + (h*h/12.0) * q_im1       
    b_coef =  2.0 + (5.0*h*h/6.0) * q_i       
    c_coef = -1.0 + (h*h/12.0) * q_ip1      
   
    rhs = np.zeros(n, dtype=float)
    rhs[0]  = - a_coef[0] * phi_a
    rhs[-1] = - c_coef[-1] * phi_b

    U_int = LU_tridiag(a_coef, b_coef, c_coef, rhs)

    U = np.zeros(N+1)
    U[0] = phi_a; U[-1] = phi_b; U[1:-1] = U_int

    exact_vals = exact_phi(x)
    err = np.max(np.abs(U - exact_vals))

    print(f"{h:8.2e} {err:12.3e} {err/h**3:12.3e} {err/h**4:12.3e} {err/h**5:12.3e}")

    LU_plots.append(U)
    x_plots.append(x)


for i, U in enumerate(LU_plots):
    plt.plot(x_plots[i], U, label=f"P={p_min+i}")
xx = np.linspace(0,1,300)
plt.plot(xx, exact_phi(xx), 'k--', label='Exact')
plt.legend(); plt.xlabel('x'); plt.ylabel('u(x)'); plt.grid(True); plt.show()
