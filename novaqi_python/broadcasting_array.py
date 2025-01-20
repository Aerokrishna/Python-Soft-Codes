import numpy as np
import csv
filename = '/prm_qtable.csv'
with open(filename, mode='r') as file:
            reader = csv.reader(file)
            columns = [[], [], []]  # Initialize three empty lists for the columns

            for row in reader:
                for i in range(3):  # Assuming there are exactly 3 columns
                    columns[i].append(float(row[i]))
#print(columns[0])
node_x = np.array(columns[0])
node_y = np.array(columns[1])

node = np.column_stack((node_x,node_y))

def is_within_tolerance(array1, array2, tolerance):
    # Reshape arrays for broadcasting: array1 -> (n1, 1, d), array2 -> (1, n2, d)
    array1_broadcast = array1[:, np.newaxis, :]
    array2_broadcast = array2[np.newaxis, :, :]
    
    # Compute the absolute differences and check if within tolerance
    within_tolerance = np.all(np.abs(array1_broadcast - array2_broadcast) <= tolerance, axis=2)
    
    # Find the indices where matches are found
    matching_indices = np.where(within_tolerance)
    matching_indices = np.column_stack(matching_indices)[:,0]
   
    print(matching_indices)
    
    # Extract the matching elements from array1
    matching_elements = array1[np.unique(matching_indices)]
    print(matching_elements)

    return matching_elements,matching_indices

tolerance = 0.1
Ipc_lim = np.array([[2,3],[10.51,9.68],[6.17,25.25],[5.92,35.48]])
result = is_within_tolerance(node, Ipc_lim, tolerance)

