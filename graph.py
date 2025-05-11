import matplotlib.pyplot as plt

def extract_values(filename):
    iterations = []
    pcost = []

    # Read the file and extract the relevant columns
    with open(filename, 'r') as file:
        for line in file:
            # Split each line into parts (columns)
            columns = line.split()
            if len(columns) > 2:
                try:
                    iterations.append(int(columns[0]))
                    pcost.append(float(columns[1]))
                except ValueError:
                    continue

    return iterations, pcost

# Function to plot the graph
def plot_iteration_vs_pcost(iterations, pcost):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, pcost, color='blue', marker='o', linestyle='-', markersize=5)

    plt.title('pcost vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('pcost')
    plt.grid(True)
    plt.show()


filename = '/Users/pradeeppatil/workspace/ECE647_Project/output.txt'  # Change this to the actual path of your output file

iterations, pcost = extract_values(filename)

print(iterations, pcost)

plot_iteration_vs_pcost(iterations, pcost)
