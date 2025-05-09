import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from mpl_toolkits.mplot3d import Axes3D

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

# Define optimization variables
x = cp.Variable((NUM_SLICES, NUM_CELLS), nonneg=True)  # PRBs allocated to each slice in each cell
p = cp.Variable((NUM_SLICES, NUM_CELLS), nonneg=True)  # Power allocated to each slice in each cell
f = cp.Variable(NUM_SLICES, nonneg=True)  # Fronthaul bandwidth allocated to each slice
c = cp.Variable(NUM_SLICES, nonneg=True)  # Compute resources allocated to each slice

# Objective function: Maximize the weighted sum-rate utility (proportional fairness)
objective = cp.Maximize(
    cp.sum([weights[i] * cp.sum(cp.log(1 + (p[i, j] * h[i, j]) / NOISE_POWER)) for i in range(NUM_SLICES) for j in range(NUM_CELLS)])
)

# Constraints
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

# Create lists to store intermediate pcost and dcost
pcost_list = []
dcost_list = []

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem with verbose output
problem.solve(solver=cp.ECOS, verbose=True)

# Print the optimized values
print("Optimized PRBs (x):", x.value)
print("Optimized Power (p):", p.value)
print("Optimized Fronthaul (f):", f.value)
print("Optimized Compute (c):", c.value)

# Compute objective function values per (i, j)
objective_values = np.zeros((NUM_SLICES, NUM_CELLS))

for i in range(NUM_SLICES):
    for j in range(NUM_CELLS):
        snr = (p.value[i, j] * h[i, j]) / NOISE_POWER
        objective_values[i, j] = weights[i] * np.log(1 + snr)

inverted_objective_values = -objective_values

# Create meshgrid for i, j
I, J = np.meshgrid(np.arange(NUM_CELLS), np.arange(NUM_SLICES))

# Plotting
fig = plt.figure(figsize=(14, 7))  # Increased figure size to accommodate two plots
fig2 = plt.figure(figsize=(14, 7))  # Increased figure size to accommodate two plots


# Plot original objective function
ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, first plot
ax1.plot_surface(I, J, objective_values)
ax1.set_xlabel('Cell Index (j)')
ax1.set_ylabel('Slice Index (i)')
ax1.set_zlabel('Objective Function Contribution')
ax1.set_title('Objective Function Value per (Slice, Cell)')

# Plot inverted objective function
ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, second plot
ax2.plot_surface(I, J, inverted_objective_values)
ax2.set_xlabel('Cell Index (j)')
ax2.set_ylabel('Slice Index (i)')
ax2.set_zlabel('Negative Objective Function Contribution')
ax2.set_title('Concave Representation of Objective Function Value per (Slice, Cell)')

def extract_values(filename):
    iterations = []
    pcost = []

    # Read the file and extract the relevant columns
    with open(filename, 'r') as file:
        for line in file:
            # Split each line into parts (columns)
            columns = line.split()
            if len(columns) > 2:
                print(columns)
                # try:
                #     # Extract iteration number from the first column and pcost from the second column
                #     iterations.append(int(columns[0]))
                #     pcost.append(float(columns[1]))
                # except ValueError:
                #     # Skip lines that do not contain valid integer or float values
                #     continue

    return iterations, pcost
extract_values('/Users/pradeeppatil/workspace/ECE647_Project/output.txt')
# ax3 = fig2.add_subplot(123, projection='3d')
# ax3.plot_surface(I, J, inverted_objective_values)
# ax3.set_xlabel('Cell Index (j)')
# ax3.set_ylabel('Slice Index (i)')
# ax3.set_zlabel('Negative Objective Function Contribution')
# ax3.set_title('Concave Representation of Objective Function Value per (Slice, Cell)')



plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

