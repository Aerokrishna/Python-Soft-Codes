import cv2
import numpy as np

width = 300
height = 200
img = cv2.imread("my_map.pgm", cv2.IMREAD_COLOR)
# print(img)

resized_image = cv2.resize(img, (width, height))
resized_image2 = cv2.resize(img, (400, 300))

cell_size = 1 # that is one cell contains 4 rows and columns of pixels

# Ensure occupancyGrid matches resized image dimensions
occupancyGrid = np.zeros((height // cell_size, width // cell_size), dtype=np.int8)

# Define color thresholds (modify as needed)
white_threshold = [225, 225, 225]  # RGB for free space
black_threshold = [80, 80, 80]     # RGB for occupied space
gray_threshold = [205, 205, 205]   # Adjust for unexplored space

for rows in range(0, height, cell_size):
    for columns in range(0, width, cell_size):
        white_count = 0
        black_count = 0
        gray_count = 0
        no_count = 0

        for i in range(cell_size):
            for j in range(cell_size):
                pixel = resized_image[i + rows][j + columns]
                # Check all colors once per 4x4 block
                if pixel[0] > white_threshold[0] and pixel[1] > white_threshold[1] and pixel[2] > white_threshold[2]:
                    white_count += 1
                elif pixel[0] < black_threshold[0] and pixel[1] < black_threshold[1] and pixel[2] < black_threshold[2]:
                    black_count += 1
                elif np.all(pixel == gray_threshold):
                    gray_count += 1
                else :
                    no_count += 1

        dominant_color = max(white_count, black_count, gray_count,no_count)
        # print(no_count)
        cell_rows = rows // cell_size
        cell_columns = columns // cell_size

        if dominant_color == white_count:
            occupancyGrid[cell_rows][cell_columns] = 0  # Free space
        elif dominant_color == black_count:
            occupancyGrid[cell_rows][cell_columns] = 100  # Occupied
        elif dominant_color == gray_count:
            occupancyGrid[cell_rows][cell_columns] = 100  # Unexplored
        else:
            occupancyGrid[cell_rows][cell_columns] = 5  # Unexplored
            # print(resized_image[rows][columns])


np.set_printoptions(threshold=np.inf)
print(occupancyGrid.shape)
print(occupancyGrid)
# print(resized_image[1][296])

cv2.imshow("image", resized_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Ensure occupancyGrid matches resized image dimensions
# occupancyGrid = np.zeros((height // cell_size, width // cell_size), dtype=np.float32)

# # Define color thresholds (modify as needed)
# threshold = 100

# white_count = 0
# black_count = 0


# for rows in range(0, height, cell_size):
#     for columns in range(0, width, cell_size):

#         for i in range(cell_size):
#             for j in range(cell_size):
#                 pixel = resized_img[i + rows][j + columns]
#                 # Check all colors once per 4x4 block
            
#                 if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
#                     white_count += 1
#                 else : 
#                     black_count += 1
                
#         dominant_color = max(white_count, black_count)
    
#         cell_rows = rows // cell_size
#         cell_columns = columns // cell_size

#         if dominant_color == white_count:
#             occupancyGrid[cell_rows][cell_columns] = 1  # Occupied
#             white_count = 0
#             black_count = 0
#         else : 
#             occupancyGrid[cell_rows][cell_columns] = 0  # Free space
#             white_count = 0
#             black_count = 0