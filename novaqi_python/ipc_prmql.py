import numpy as np
import time
import numpy as np

start_time = time.time()
# Initialize arrays
sample_x = np.zeros(1000, dtype='float32')  # Example array with zeros
Ipc_lim = np.random.uniform(low=0.0, high=10.0, size=30)

# Step 1: Identify matching elements
mask = np.isin(Ipc_lim, sample_x)

# Step 2: Extract matching elements
matching_elements = Ipc_lim[mask]

# Assume we have an existing array to which we want to append, or create a new one
# If it does not exist:
matches_array = np.array([], dtype='float32')

# Step 3: Append matching elements
matches_array = np.concatenate((matches_array, matching_elements))

print(matches_array)


# End timer
end_time = time.time()

# Calculate total runtime
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime} seconds")
