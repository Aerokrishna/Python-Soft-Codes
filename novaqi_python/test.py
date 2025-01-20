import numpy as np

def is_within_tolerance(array1, array2, tolerance):
    # Reshape arrays for broadcasting: array1 -> (n1, 1, d), array2 -> (1, n2, d)
    array1_broadcast = array1[:, np.newaxis, :]
    array2_broadcast = array2[np.newaxis, :, :]
    
    # Compute the absolute differences and check if within tolerance
    within_tolerance = np.all(np.abs(array1_broadcast - array2_broadcast) <= tolerance, axis=2)
    
    # Find the indices where matches are found
    matching_indices = np.where(within_tolerance)
    
    # Extract the matching elements from array1
    matching_elements = array1[np.unique(matching_indices[0])]
    
    return matching_elements

array1 = np.array([[0, 0], [1, 1], [2, 11]])
array2 = np.array([[1, 1], [3, 3], [0, 2],[0,1],[2,11]])
tolerance = 0.5

result = is_within_tolerance(array1, array2, tolerance)
print(result)
