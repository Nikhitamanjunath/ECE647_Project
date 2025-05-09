import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
 
# ======================
# 1. SYSTEM MODEL SETUP
# ======================
 
# Network parameters
num_slices = 3  # eMBB, URLLC, mMTC
num_cells = 4    # Number of base stations
num_RBs = 100    # Total resource blocks per cell
max_power = 20   # Max transmit power per cell (Watts)
fronthaul_cap = 1e9  # 1 Gbps fronthaul capacity
compute_cap = 100    # 100 vCPU units
 
# Slice characteristics (weights, requirements)
slice_weights = {'eMBB': 0.5, 'URLLC': 0.8, 'mMTC': 0.3}
slice_latency_req = {'eMBB': 10, 'URLLC': 1, 'mMTC': 100}  # ms
 
# Channel model (random SINR values for demonstration)
def generate_channel_conditions():
    # SINR matrix: slices x cells
    return np.random.exponential(scale=10, size=(num_slices, num_cells))
 
# ======================
# 2. OPTIMIZATION PROBLEM
# ======================
 
def solve_slicing_problem(SINR, lambda_fronthaul=0, lambda_compute=0):
    """
    Solves the convex resource slicing problem for given dual variables
    Returns: RB allocation, power allocation, fronthaul usage
    """
    # Variables
    x = cp.Variable((num_slices, num_cells), nonneg=True)  # RB allocation
    p = cp.Variable((num_slices, num_cells), nonneg=True)  # Power allocation
    f = cp.Variable(num_slices, nonneg=True)              # Fronthaul usage
    c = cp.Variable(num_slices, nonneg=True)              # Compute usage
    # Objective: Weighted sum-rate minus penalty for constraints
    objective_terms = []
    for i in range(num_slices):
        for j in range(num_cells):
            rate = cp.log(1 + SINR[i,j] * p[i,j] / (x[i,j] + 1e-6))  # Avoid division by zero
            objective_terms.append(slice_weights[i] * x[i,j] * rate)
    objective = cp.Maximize(
        cp.sum(objective_terms) - 
        lambda_fronthaul * cp.sum(f) - 
        lambda_compute * cp.sum(c)
    )
    # Constraints
    constraints = []
    # 1. Per-cell constraints
    for j in range(num_cells):
        # Total power constraint
        constraints.append(cp.sum(p[:, j]) <= max_power)
        # Total RB constraint
        constraints.append(cp.sum(x[:, j]) <= num_RBs)
    # 2. Slice constraints
    for i in range(num_slices):
        # Fronthaul proportional to traffic
        constraints.append(f[i] == 1e6 * cp.sum(x[i, :]))  # 1 Mbps per RB
        # Compute proportional to processing
        constraints.append(c[i] == 0.1 * cp.sum(x[i, :] * SINR[i, :]))  # Higher SINR needs more processing
        # Latency constraints (simplified model)
        processing_delay = 5 / (c[i] + 1e-6)  # ms
        fronthaul_delay = (f[i] * 8) / (fronthaul_cap / num_slices)  # ms
        constraints.append(processing_delay + fronthaul_delay <= slice_latency_req[i])
    # Solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    return x.value, p.value, f.value, c.value
 
# ======================
# 3. DUAL DECOMPOSITION ALGORITHM
# ======================
 
def dual_decomposition(max_iter=100):
    """Main optimization loop with dual updates"""
    # Initialize dual variables
    lambda_fronthaul = 0
    lambda_compute = 0
    # Step sizes
    alpha_f = 1e-8
    alpha_c = 1e-6
    # Storage for results
    history = {
        'fronthaul_usage': [],
        'compute_usage': [],
        'lambda_f': [],
        'lambda_c': [],
        'objective': []
    }
    SINR = generate_channel_conditions()
    for it in tqdm(range(max_iter)):
        # Solve subproblem with current dual variables
        x, p, f, c = solve_slicing_problem(SINR, lambda_fronthaul, lambda_compute)
        # Store results
        total_fronthaul = np.sum(f)
        total_compute = np.sum(c)
        history['fronthaul_usage'].append(total_fronthaul)
        history['compute_usage'].append(total_compute)
        history['lambda_f'].append(lambda_fronthaul)
        history['lambda_c'].append(lambda_compute)
        # Dual variable updates
        lambda_fronthaul += alpha_f * (total_fronthaul - fronthaul_cap)
        lambda_compute += alpha_c * (total_compute - compute_cap)
        # Project onto non-negative orthant
        lambda_fronthaul = max(0, lambda_fronthaul)
        lambda_compute = max(0, lambda_compute)
        # Calculate objective (primal problem without penalties)
        _, _, obj = solve_slicing_problem(SINR, 0, 0)
        history['objective'].append(obj)
    return history, (x, p, f, c)
 
# ======================
# 4. SIMULATION AND PLOTTING
# ======================
 
def run_simulation():
    # Run optimization
    history, allocations = dual_decomposition(max_iter=50)
    x, p, f, c = allocations
    # Plot results
    plt.figure(figsize=(15, 10))
    # Plot 1: Resource usage
    plt.subplot(2, 2, 1)
    plt.plot(history['fronthaul_usage'], label='Fronthaul Usage')
    plt.axhline(y=fronthaul_cap, color='r', linestyle='--', label='Capacity')
    plt.xlabel('Iteration')
    plt.ylabel('Fronthaul (Mbps)')
    plt.title('Fronthaul Usage Convergence')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(history['compute_usage'], label='Compute Usage')
    plt.axhline(y=compute_cap, color='r', linestyle='--', label='Capacity')
    plt.xlabel('Iteration')
    plt.ylabel('Compute (vCPU)')
    plt.title('Compute Usage Convergence')
    plt.legend()
    # Plot 2: Dual variables
    plt.subplot(2, 2, 3)
    plt.plot(history['lambda_f'], label='Fronthaul Dual')
    plt.plot(history['lambda_c'], label='Compute Dual')
    plt.xlabel('Iteration')
    plt.ylabel('Dual Variable Value')
    plt.title('Dual Variable Convergence')
    plt.legend()
    # Plot 3: RB allocation (final iteration)
    plt.subplot(2, 2, 4)
    slice_names = ['eMBB', 'URLLC', 'mMTC']
    for i in range(num_slices):
        plt.bar(np.arange(num_cells) + 0.2*i, x[i, :], width=0.2, label=slice_names[i])
    plt.xlabel('Cell Index')
    plt.ylabel('RB Allocation')
    plt.title('Final RB Allocation Per Cell')
    plt.legend()
    plt.tight_layout()
    plt.show()
 
    # Print final allocations
    print("\nFinal Allocations:")
    print(f"Total fronthaul used: {np.sum(f):.2f} / {fronthaul_cap} Mbps")
    print(f"Total compute used: {np.sum(c):.2f} / {compute_cap} vCPUs")
    print("\nPer-slice allocations:")
    for i in range(num_slices):
        print(f"{slice_names[i]}:")
        print(f"  RBs: {np.sum(x[i, :]):.1f}, Power: {np.sum(p[i, :]):.2f}W")
        print(f"  Fronthaul: {f[i]:.2f}Mbps, Compute: {c[i]:.2f}vCPUs")
 
# Run the simulation
if __name__ == "__main__":
    run_simulation()