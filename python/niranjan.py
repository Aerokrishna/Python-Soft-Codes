import cv2
import numpy as np
widht=640
cap = cv2.VideoCapture("/dev/video0")  # 0 corresponds to the default camera (change if necessary)
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, (widht, 480))

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x, y = 320, 240 
    hsv_value = hsv_frame[y, x]

    # Extract individual components
    hue = hsv_value[0]
    saturation = hsv_value[1]
    value = hsv_value[2]

    # Print the individual HSV components
    print("Hue:", hue)
    print("Saturation:", saturation)
    print("Value:", value)
    center=(x,y)
    cv2.circle(frame, center, 8, (0, 0, 0),-1)

        # Display the result with the original frame
    cv2.imshow('Color Segmentation', frame)
    #cv2.imshow('Color', color_mask)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
