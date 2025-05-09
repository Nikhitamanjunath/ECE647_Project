import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Network parameters
NUM_SLICES = 3  # eMBB, URLLC, mMTC
NUM_CELLS = 1
PRB_TOTAL = 100  # Total Physical Resource Blocks
P_MAX = 20       # Max transmit power (Watts)
F_MAX = 10e9     # Fronthaul capacity (10 Gbps)
C_MAX = 50       # Compute capacity (vCPUs)
NOISE_POWER = 1e-9  # σ^2 (noise power)

# Slice parameters (weights, channel gains)
weights = {'eMBB': 1, 'URLLC': 5, 'mMTC': 0.5}
h = np.abs(np.random.randn(NUM_SLICES, NUM_CELLS))  # Channel gains |h_{i,j}|^2

# URLLC-specific parameters
KAPPA = 0.5  # cycles/bit (processing)
D = 1000     # packet size (bits)

def solve_primal_problem():
    """Solve the primal convex optimization problem"""
    # Optimization variables
    x = cp.Variable((NUM_SLICES, NUM_CELLS), nonneg=True)  # PRBs
    p = cp.Variable((NUM_SLICES, NUM_CELLS), nonneg=True)  # Power
    f = cp.Variable(NUM_SLICES, nonneg=True)               # Fronthaul
    c = cp.Variable(NUM_SLICES, nonneg=True)               # Compute
    
    # Objective: Weighted sum-rate (proportional fairness)
    objective = 0
    for i in range(NUM_SLICES):
        for j in range(NUM_CELLS):
            sinr = (p[i,j] * h[i,j]) / NOISE_POWER  # Approximated SINR
            objective += weights[list(weights.keys())[i]] * cp.log(1 + sinr)
    
    # Constraints
    constraints = [
        cp.sum(p) <= P_MAX,                     # Total power limit
        cp.sum(x) <= PRB_TOTAL,                 # Total PRB limit
        cp.sum(f) <= F_MAX,                     # Fronthaul capacity
        cp.sum(c) <= C_MAX,                     # Compute capacity
        (KAPPA/c[1]) + (D/f[1]) <= 1e-3,       # URLLC latency (1ms)
        x >= 0, p >= 0, f >= 0, c >= 0          # Non-negativity
    ]
    
    # Form and solve problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.ECOS)
    
    return x.value, p.value, f.value, c.value

def dual_decomposition(max_iters=100, tol=1e-3):
    """Dual decomposition algorithm for distributed optimization"""
    # Initialize dual variables
    lambda_f = 1.0  # Fronthaul dual variable
    lambda_c = 1.0  # Compute dual variable
    alpha = 0.1     # Step size
    
    # History tracking
    f_history = []
    c_history = []
    lambda_history = []
    
    for k in range(max_iters):
        # Solve per-slice subproblems in parallel
        f_alloc = np.zeros(NUM_SLICES)
        c_alloc = np.zeros(NUM_SLICES)
        
        for i in range(NUM_SLICES):
            # Define variables for slice i
            x_i = cp.Variable(nonneg=True)
            p_i = cp.Variable(nonneg=True)
            f_i = cp.Variable(nonneg=True)
            c_i = cp.Variable(nonneg=True)
            
            # Subproblem objective
            sinr = (p_i * h[i,0]) / NOISE_POWER
            obj = weights[list(weights.keys())[i]] * cp.log(1 + sinr) - \
                  lambda_f * f_i - lambda_c * c_i
            
            # Subproblem constraints
            constraints = [
                p_i <= P_MAX/NUM_SLICES,  # Local power limit
                x_i <= PRB_TOTAL/NUM_SLICES,  # Local PRB limit
                f_i <= F_MAX,  # Individual fronthaul limit
                c_i <= C_MAX,  # Individual compute limit
            ]
            
            # Add URLLC latency constraint if applicable
            if i == 1:  # Assuming URLLC is slice 1
                constraints += [(KAPPA/c_i) + (D/f_i) <= 1e-3]
            
            # Solve subproblem
            subproblem = cp.Problem(cp.Maximize(obj), constraints)
            subproblem.solve()
            
            f_alloc[i] = f_i.value
            c_alloc[i] = c_i.value
        
        # Update dual variables via gradient ascent
        lambda_f += alpha * (sum(f_alloc) - F_MAX)
        lambda_c += alpha * (sum(c_alloc) - C_MAX)
        
        # Project onto non-negative orthant
        lambda_f = max(0, lambda_f)
        lambda_c = max(0, lambda_c)
        
        # Store history
        f_history.append(f_alloc.copy())
        c_history.append(c_alloc.copy())
        lambda_history.append((lambda_f, lambda_c))
        
        # Check convergence
        if k > 0 and \
           np.abs(lambda_history[-1][0] - lambda_history[-2][0]) < tol and \
           np.abs(lambda_history[-1][1] - lambda_history[-2][1]) < tol:
            break
    
    return f_history, c_history, lambda_history

def plot_results(f_history, c_history, lambda_history):
    """Plot convergence results"""
    plt.figure(figsize=(15, 5))
    
    # Plot fronthaul allocation
    plt.subplot(1, 3, 1)
    for i in range(NUM_SLICES):
        plt.plot([f[i] for f in f_history], label=f'Slice {i}')
    plt.xlabel('Iterations')
    plt.ylabel('Fronthaul Allocation')
    plt.title('Fronthaul Allocation Convergence')
    plt.legend()
    
    # Plot compute allocation
    plt.subplot(1, 3, 2)
    for i in range(NUM_SLICES):
        plt.plot([c[i] for c in c_history], label=f'Slice {i}')
    plt.xlabel('Iterations')
    plt.ylabel('Compute Allocation')
    plt.title('Compute Allocation Convergence')
    plt.legend()
    
    # Plot dual variables
    plt.subplot(1, 3, 3)
    plt.plot([l[0] for l in lambda_history], label='λ_f (Fronthaul)')
    plt.plot([l[1] for l in lambda_history], label='λ_c (Compute)')
    plt.xlabel('Iterations')
    plt.ylabel('Dual Variable Value')
    plt.title('Dual Variable Convergence')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Run simulations
print("Solving primal problem...")
x_opt, p_opt, f_opt, c_opt = solve_primal_problem()
print("Optimal allocations:")
print(f"PRBs: {x_opt}")
print(f"Power: {p_opt}")
print(f"Fronthaul: {f_opt}")
print(f"Compute: {c_opt}")

print("\nRunning dual decomposition...")
f_history, c_history, lambda_history = dual_decomposition()
plot_results(f_history, c_history, lambda_history)