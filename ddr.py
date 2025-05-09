import numpy as np

# Problem Parameters (Adjust these as needed)
NUM_SLICES = 2  # Number of slices
NUM_CELLS = 3  # Number of cells
F_MAX = 10  # Maximum fronthaul capacity
C_MAX = 20  # Maximum compute capacity
P_MAX = 100 # Maximum Power
NOISE_POWER = 1e-9
KAPPA = 1000
D = 100

# Channel gains (h[slice, cell]) -  Random example, replace with your actual data
h = np.random.rand(NUM_SLICES, NUM_CELLS)
# Slice weights
weights = np.array([1, 1])  # Equal weights for simplicity

# URLLC Slices
URLLC_SLICES = [0]

# Dual Decomposition Parameters
lambda_f = 1.0  # Initial dual variable for fronthaul
lambda_c = 1.0  # Initial dual variable for compute
alpha = 0.1  # Step size
epsilon = 1e-4  # Convergence tolerance
K_max = 100  # Maximum iterations

# Initialize variables to store results
x = np.zeros((NUM_SLICES, NUM_CELLS))
p = np.zeros((NUM_SLICES, NUM_CELLS))
f = np.zeros(NUM_SLICES)
c = np.zeros(NUM_SLICES)

def slice_subproblem(slice_index, lambda_f_val, lambda_c_val):
    """
    Solves the subproblem for a given slice.  This is where the optimization happens.
    For simplicity, I'm using a gradient ascent-like approach here
    because a full-fledged optimization library (like scipy.optimize)
    would add a lot of complexity to this *simple* example.  In a
    real-world scenario, you'd use a proper solver.

    Args:
        slice_index: The index of the slice (0, 1, ...).
        lambda_f_val: Current value of the fronthaul dual variable.
        lambda_c_val: Current value of the compute dual variable.

    Returns:
        Tuple: (x_opt, p_opt, f_opt, c_opt, objective_value)
               Returns None, None, None, None, -np.inf if it cannot find a solution.
    """
    num_cells = h.shape[1]
    x_opt = np.zeros(num_cells)
    p_opt = np.zeros(num_cells)
    f_opt = 0.0
    c_opt = 0.0
    objective_value = -np.inf # Default to negative infinity in case of issues.

    # Very basic, non-optimized allocation strategy for demonstration.
    #  * Allocate power and PRBs equally across cells for this slice.
    #  * Set f and c based on URLLC constraint if applicable, otherwise, allocate a small amount.

    p_total_slice = P_MAX / NUM_SLICES
    prb_total_slice = 50 / NUM_SLICES # Assume 50 total PRBs.

    for cell in range(num_cells):
        p_opt[cell] = p_total_slice / num_cells
        x_opt[cell] = prb_total_slice / num_cells

    if slice_index in URLLC_SLICES:
        c_opt = (1e-3) / KAPPA  # Minimum c to meet latency if f is small
        f_opt = (1e-3) / D
    else:
        f_opt = 1.0  # Just allocate a small amount
        c_opt = 1.0

    # Calculate a simplified objective (the actual objective calculation
    # would normally be part of a proper optimization routine).
    rate_sum = 0
    for j in range(num_cells):
        rate_sum += np.log2(1 + (p_opt[j] * h[slice_index, j]) / NOISE_POWER)
    objective_value = weights[slice_index] * rate_sum - lambda_f_val * f_opt - lambda_c_val * c_opt

    return x_opt, p_opt, f_opt, c_opt, objective_value



# Main Dual Decomposition Loop
for k in range(K_max):
    print(f"Iteration {k + 1}")
    f_sum = 0
    c_sum = 0

    # 1. Solve Subproblems for each slice
    for i in range(NUM_SLICES):
        x_slice, p_slice, f_slice, c_slice, obj_value = slice_subproblem(i, lambda_f, lambda_c)
        if x_slice is not None: # Check for a valid solution
            x[i, :] = x_slice
            p[i, :] = p_slice
            f[i] = f_slice
            c[i] = c_slice
            f_sum += f_slice
            c_sum += c_slice
        else:
            print(f"Slice {i} subproblem failed.")
            # Handle the failure.  For this simple example, we'll just use the previous values
            # In a real implementation you might want to re-initialize the subproblem.


    # 2. Update Dual Variables
    lambda_f_new = max(0, lambda_f + alpha * (f_sum - F_MAX))
    lambda_c_new = max(0, lambda_c + alpha * (c_sum - C_MAX))

    # 3. Check for Convergence
    if abs(lambda_f_new - lambda_f) < epsilon and abs(lambda_c_new - lambda_c) < epsilon:
        print("Converged!")
        break

    lambda_f = lambda_f_new
    lambda_c = lambda_c_new

    print(f"  lambda_f = {lambda_f:.4f}, lambda_c = {lambda_c:.4f}")

else:
    print("Maximum iterations reached.")

# Print Results
print("\n--- Results ---")
print("Allocated PRBs (x):")
print(x)
print("Allocated Power (p):")
print(p)
print("Fronthaul Capacity (f):")
print(f)
print("Compute Capacity (c):")
print(c)
print(f"Final lambda_f: {lambda_f:.4f}, Final lambda_c: {lambda_c:.4f}")
