import csv

# save

# Assuming your matrix is a 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Save the matrix to a CSV file
with open('matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(matrix)


# use

# Load the matrix from the CSV file
with open('matrix.csv', 'r') as f:
    reader = csv.reader(f)
    loaded_matrix = [list(map(int, row)) for row in reader]

print(loaded_matrix)
