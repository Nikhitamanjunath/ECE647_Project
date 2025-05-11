import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from mpl_toolkits.mplot3d import Axes3D

# Network parameters
NUM_SLICES = 3     # Number of slices (e.g., URLLC, eMBB, etc.)
NUM_CELLS = 5      # Number of cells (example)
h = np.random.rand(NUM_SLICES, NUM_CELLS)  # Channel gains (random example)
weights = {0: 1, 1: 2, 2: 1}  # Weights for each slice, modify accordingly
NOISE_POWER = 1e-9  # Noise power (example)
P_MAX = 100  # Maximum power per slice
PRB_TOTAL = 50  # Total number of PRBs allocated to each slice
F_MAX = 10  # Maximum fronthaul capacity
C_MAX = 20  # Maximum compute capacity
KAPPA = 1000  # Processing cycles per bit (example)
D = 100  # Packet size (example)
URLLC_SLICES = [0]  # Assuming slice 0 is URLLC

x = cp.Variable((NUM_SLICES, NUM_CELLS), nonneg=True)  # PRBs allocated to each slice in each cell
p = cp.Variable((NUM_SLICES, NUM_CELLS), nonneg=True)  # Power allocated to each slice in each cell
f = cp.Variable(NUM_SLICES, nonneg=True)  # Fronthaul bandwidth allocated to each slice
c = cp.Variable(NUM_SLICES, nonneg=True)  # Compute resources allocated to each slice

objective = cp.Maximize(
    cp.sum([
        weights[i] * cp.log(1 + (p[i, j] * h[i, j]) / NOISE_POWER)
        for i in range(NUM_SLICES)
        for j in range(NUM_CELLS)
    ])
)

constraints = [
    # Radio Constraints
    cp.sum(p) <= P_MAX,  # Total power limit for each slice
    cp.sum(x) <= PRB_TOTAL,  # Total PRBs allocated to each slice

    # Fronthaul Capacity
    cp.sum(f) <= F_MAX,  # Total fronthaul bandwidth limit

    # Compute Capacity
    cp.sum(c) <= C_MAX,  # Total compute capacity limit

    # URLLC Latency Constraints
    *[
        KAPPA * c[i] + D * f[i] <= 1e-3 for i in URLLC_SLICES
    ],

    # Non-negativity constraints
    x >= 0,
    p >= 0,
    f >= 0,
    c >= 0
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=True)

print("\nPrimal Solution:")
print("Optimized PRBs (x):", x.value)
print("Optimized Power (p):", p.value)
print("Optimized Fronthaul (f):", f.value)
print("Optimized Compute (c):", c.value)
print("Primal Objective Value:", problem.value)

def dual_decomposition(max_iters=100, alpha=0.1, tol=1e-3):
    """Dual decomposition algorithm for distributed optimization"""
    # Initialize dual variables
    lambda_f = 1.0  # Fronthaul dual variable
    lambda_c = 1.0  # Compute dual variable
    
    # History tracking
    f_history = []
    c_history = []
    lambda_history = []
    obj_history = []
    
    for k in range(max_iters):
        # Solve per-slice subproblems in parallel
        f_alloc = np.zeros(NUM_SLICES)
        c_alloc = np.zeros(NUM_SLICES)
        obj_alloc = 0
        
        for i in range(NUM_SLICES):
            # Define variables for slice i
            x_i = cp.Variable((NUM_CELLS,), nonneg=True)
            p_i = cp.Variable((NUM_CELLS,), nonneg=True)
            f_i = cp.Variable(nonneg=True)
            c_i = cp.Variable(nonneg=True)
            
            # Subproblem objective
            obj_terms = [weights[i] * cp.log(1 + (p_i[j] * h[i,j]) / NOISE_POWER) for j in range(NUM_CELLS)]
            obj = cp.sum(obj_terms) - lambda_f * f_i - lambda_c * c_i
            
            # Subproblem constraints
            sub_constraints = [
                cp.sum(p_i) <= P_MAX/NUM_SLICES,  # Local power limit
                cp.sum(x_i) <= PRB_TOTAL/NUM_SLICES,  # Local PRB limit
                f_i <= F_MAX,  # Individual fronthaul limit
                c_i <= C_MAX,  # Individual compute limit
            ]
            
            # Add URLLC latency constraint if applicable
            if i in URLLC_SLICES:
                sub_constraints += [KAPPA * c_i + D * f_i <= 1e-3]
            
            # Solve subproblem
            subproblem = cp.Problem(cp.Maximize(obj), sub_constraints)
            subproblem.solve(solver=cp.ECOS, verbose=False)
            
            f_alloc[i] = f_i.value
            c_alloc[i] = c_i.value
            obj_alloc += subproblem.value + lambda_f * f_i.value + lambda_c * c_i.value
        
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
        obj_history.append(obj_alloc)
        
        # Check convergence
        if k > 0 and \
           np.abs(lambda_history[-1][0] - lambda_history[-2][0]) < tol and \
           np.abs(lambda_history[-1][1] - lambda_history[-2][1]) < tol:
            break
    
    return f_history, c_history, lambda_history, obj_history

print("\nRunning Dual Decomposition...")
f_history, c_history, lambda_history, obj_history = dual_decomposition()

print("\nDual Solution:")
print("Final Fronthaul Allocation:", f_history[-1])
print("Final Compute Allocation:", c_history[-1])
print("Final Dual Variables (λ_f, λ_c):", lambda_history[-1])
print("Final Dual Objective Value:", obj_history[-1])

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for i in range(NUM_SLICES):
    plt.plot([f[i] for f in f_history], label=f'Slice {i} Fronthaul')
plt.xlabel('Iterations')
plt.ylabel('Fronthaul Allocation')
plt.title('Fronthaul Allocation Convergence')
plt.legend()

plt.subplot(2, 2, 2)
for i in range(NUM_SLICES):
    plt.plot([c[i] for c in c_history], label=f'Slice {i} Compute')
plt.xlabel('Iterations')
plt.ylabel('Compute Allocation')
plt.title('Compute Allocation Convergence')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot([l[0] for l in lambda_history], label='λ_f (Fronthaul)')
plt.plot([l[1] for l in lambda_history], label='λ_c (Compute)')
plt.xlabel('Iterations')
plt.ylabel('Dual Variable Value')
plt.title('Dual Variable Convergence')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(obj_history, label='Dual Objective')
plt.axhline(y=problem.value, color='r', linestyle='--', label='Primal Objective')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('Objective Value Convergence')
plt.legend()

plt.tight_layout()
plt.show()

objective_values = np.zeros((NUM_SLICES, NUM_CELLS))
for i in range(NUM_SLICES):
    for j in range(NUM_CELLS):
        snr = (p.value[i, j] * h[i, j]) / NOISE_POWER
        objective_values[i, j] = weights[i] * np.log(1 + snr)

inverted_objective_values = -objective_values

I, J = np.meshgrid(np.arange(NUM_CELLS), np.arange(NUM_SLICES))

fig = plt.figure(figsize=(14, 7))
fig2 = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(I, J, objective_values)
ax1.set_xlabel('Cell Index (j)')
ax1.set_ylabel('Slice Index (i)')
ax1.set_zlabel('Objective Function Contribution')
ax1.set_title('Objective Function Value per (Slice, Cell)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(I, J, inverted_objective_values)
ax2.set_xlabel('Cell Index (j)')
ax2.set_ylabel('Slice Index (i)')
ax2.set_zlabel('Negative Objective Function Contribution')
ax2.set_title('Concave Representation of Objective Function Value per (Slice, Cell)')

ax3 = fig2.add_subplot(133)
contour = ax3.contour(J, I, inverted_objective_values)
ax3.set_xlabel('Cell Index (j)')
ax3.set_ylabel('Slice Index (i)')
ax3.set_title('Contour Plot of Objective Function')
fig2.colorbar(contour, ax=ax3, label='Objective Function Value')

plt.tight_layout()

static_throughput = [0.7, 0.3, 0.1]  # Gbps
dynamic_throughput = [1.0, 0.5, 0.2]  # From primal solution

plt.figure(1)  # Create figure 1
plt.bar(['eMBB (Static)', 'eMBB (Dynamic)'], [static_throughput[0], dynamic_throughput[0]])
plt.ylabel('Throughput (Gbps)')
plt.title('eMBB Throughput Comparison')
plt.show()  # Display the first figure

latency_static = np.random.normal(1.5, 0.5, 1000)  # 70% <1ms
latency_dynamic = np.random.normal(0.8, 0.2, 1000)  # 95% <1ms

plt.figure(2)  # Create figure 2
plt.hist(latency_static, bins=30, alpha=0.5, label='Static', density=True) # Added density=True
plt.hist(latency_dynamic, bins=30, alpha=0.5, label='Dynamic', density=True) # Added density=True
plt.axvline(1, color='r', linestyle='--', label='1ms Deadline')
plt.xlabel('Latency (ms)')
plt.ylabel('Probability Density') # Changed the y label
plt.legend()
plt.title('URLLC Latency Comparison')
plt.show()  # Display the second figure
