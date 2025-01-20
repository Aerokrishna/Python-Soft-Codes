import numpy as np
import csv
import time

# Specify the file name
filename = '/home/veejay/python/prm_qtable.csv'

# Reading the contents of the CSV file
with open(filename, mode='r') as file:
    reader = csv.reader(file)
    columns = [[], [], []]  # Initialize three empty lists for the columns

    for row in reader:
        for i in range(3):  # Assuming there are exactly 3 columns
            columns[i].append(float(row[i]))

node_x = np.array(columns[0])
node_y = np.array(columns[1])
QValue = np.array(columns[2])

start_time = time.time()

# Initialize arrays
Ipc_lim = np.random.uniform(low=0.0, high=10.0, size=30)
Ipc_lim = np.round(Ipc_lim, 2)

# Step 1: Identify matching elements
matching_elements = np.intersect1d(Ipc_lim, node_x)

# Find the indices of the matching elements in node_x
matching_indices_node_x = np.where(np.isin(node_x, matching_elements))[0]

# Print the matching indices in node_x
print("Indices of matching elements in node_x:", matching_indices_node_x)

# End timer
end_time = time.time()

# Calculate total runtime
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime} seconds")
