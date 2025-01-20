import cv2
import numpy as np
import matplotlib.pyplot as plt

width = 80
height = 80 # width/height in pixels should be same asd width/height in meters

map_img = cv2.imread("real_map4.png", cv2.IMREAD_COLOR)
# grey_map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

resized_img = cv2.resize(map_img, (width, height))

cell_size = 2 # that is one cell contains 4 rows and columns of pixels

# Ensure occupancyGrid matches resized image dimensions
occupancyGrid = np.zeros((height // cell_size, width // cell_size), dtype=np.int8)

# Define color thresholds (modify as needed)
threshold = 100

map_resolution = 20 # pixels per meter

for rows in range(0, height, cell_size):
    for columns in range(0, width, cell_size):

        white_count = 0
        black_count = 0

        for i in range(cell_size):
            for j in range(cell_size):
                pixel = resized_img[i + rows][j + columns]
                # Check all colors once per 4x4 block
               
                if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
                    white_count += 1
                else : 
                    black_count += 1
                
        dominant_color = max(white_count, black_count)
        # print(no_count)
        cell_rows = rows // cell_size
        cell_columns = columns // cell_size

        if dominant_color == white_count:
            occupancyGrid[cell_rows][cell_columns] = 100  # Occupied
        else : 
            occupancyGrid[cell_rows][cell_columns] = 0  # Free space

np.set_printoptions(threshold=np.inf)
print(occupancyGrid.shape)
print(occupancyGrid)
cv2.imshow("image", map_img)
cv2.waitKey(0)
cv2.destroyAllWindows()