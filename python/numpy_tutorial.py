import numpy as np

np.array([1,2,3,4]) # 1D array
my2d_array = np.array([[2,1,2],
          [1,3,5],
          [4,6,7]]) # 2D array
np.empty((2,2),dtype="int8") # creates a matrix of given shape with random vlaues
np.zeros((3,3),dtype="int32") # creates a matrix of given shape with 0s
np.full((3,3),5) # creates a matirx with 5s
np.reshape(np.array([[1,2,3],[5,6,7]]),(3,2)) # reshapes the array into desired shape given that the number of elements is same
np.arange(start=0,stop=2,step=0.2) # returns evenly spaced values within a given interval start and stop
np.linspace(start=1,stop=2,num=5) # returns num number of values between start and stop
my2d_array.flatten() # returns flat version of a 2d error

np.empty_like(my2d_array) # returns an array with random values of same shape as provided array
np.ones_like(my2d_array)
np.zeros_like(my2d_array)

x = np.array([0,1,2,3])
y = np.array([0,4,6,3])
X,Y = np.meshgrid(x,y) 
print(X)
print(Y)                # takes coordinates vectors x and y and produces 2D arrays 

np.identity(3) # gives an identity matrix
np.column_stack(x,y) # takes 1d arrays and stacks them as columns in a 2d array

np.unique(my2d_array,axis = 0) # searches the rows and outputs the matrix with unique rows
np.append() # V IMP

with ThreadPoolExecutor() as executor:

executor.map(lambda k: self.compute_radius(k, index, ranges, del_theta), range(start, end))) # goes through a semi cire to find all the laser points and gets the radius of each point

# np.append

# Define a 2D array
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# Append a new row (along axis 0)
new_row = np.array([[7, 8, 9]])
new_array = np.append(original_array, new_row, axis=0) # to append along the row

# Append a new column (along axis 1)
new_column = np.array([[7], [8]])
new_array_with_column = np.append(original_array, new_column, axis=1) # to append along the column

np.append(original_array,new_array,new_column)

# euclidian norm
vector = np.array([1,2,3])
np.linalg.norm(vector) # V IMP 
# returns sqrt of the sqaured sum 
vector2 = np.array([4,5,6])
np.linalg.norm(vector2 - vector) # returns euclidian distance btw the 2 vectors


# Example array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Clip values to the range [3, 7]
clipped_arr = np.clip(arr, 3, 7)

print(clipped_arr)

A = 1
reps = 5
np.tile(A,reps) # constructs an array by repeating the given element A, for reps number of times
#A can be any array or element too, it forms a linear 1d array unless specified
np.tile(A,(reps,1)) # repeats along a row
np.tile(A,(1,reps)) # repeats along column

np.reshape(A,(1,2)) # reshapes A to specific size
