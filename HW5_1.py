# Imports
import numpy as np

# Constants
tol=1e-12 
max_iter=200
x0 = [0, 0]

# G function (since we will need to call it on a loop, breaks a list into floats then back)
def G(x):
    x1, x2 = x
    g1 = (x1**2 + x2**2 + 8) / 10
    g2 = (x1 * x2**2 + x1 + 8) / 10
    return np.array([g1, g2], dtype=float)

# Main iteration loop
def fixed_point_iteration(G, x0):
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        x_new = G(x)
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        
        if diff < tol:
            return x_new, i+1, diff
        
        x = x_new

# Call solver
x_star, iterations, final_diff = fixed_point_iteration(G, x0)
# Compute residual
residual = np.linalg.norm(G(x_star) - x_star, ord=np.inf)

#Print statements
print("Approximate fixed point:")
print(x_star)
print(f"\nIterations: {iterations}")
print(f"Final difference: {final_diff}")
# Residual
residual = np.linalg.norm(G(x_star) - x_star, ord=np.inf)
print(f"Residual ||G(x*) - x*|| = {residual}")
