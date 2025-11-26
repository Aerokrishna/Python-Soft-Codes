import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import struct

# Adjust the port to match yours (e.g., COM3 on Windows, /dev/ttyACM0 or /dev/ttyUSB0 on Linux)
PORT = '/dev/ttyACM0'
BAUD = 115200

# Link lengths
L1, L2, L3 = 12, 12, 9

def inverse_kinematics(x, y, z, phi, yaw):
    # hip rotation
    hip = np.atan2(y, x)

    r = np.sqrt(x**2 + y**2)
    print("r ", r)

    """base lies in xy plane. z axis is the hip rotation in base frame"""
    # Wrist position (subtract link3 along phi direction)
    wx = r - L3*np.cos(phi)
    wy = z - L3*np.sin(phi)

    # Distance from base to wrist
    d = np.sqrt(wx**2 + wy**2)

    # Law of cosines for elbow
    cos_elbow = (d**2 - L1**2 - L2**2)/(2*L1*L2)
    cos_elbow = np.clip(cos_elbow, -1, 1)   # numerical safety
    elbow = -(np.arccos(cos_elbow))            # pick elbow-down solution

    # Shoulder
    k1 = L1 + L2*np.cos(elbow)
    k2 = L2*np.sin(elbow)
    shoulder = np.arctan2(wy, wx) - np.arctan2(k2, k1)

    # Wrist
    wrist = phi - (shoulder + elbow)

    return int(np.degrees(hip)), int(np.degrees(shoulder)), int(np.degrees(elbow)), int(np.degrees(wrist)), int(np.degrees(yaw))

# Example usage
target_pos = (10.0, 20.0, 0.0)
target_pos_l = (10.0, -20.0, 0.0)

desired_phi = np.deg2rad(-90)  
yaw = np.deg2rad(0)  

r = np.sqrt(target_pos[0]**2 + target_pos[1]**2)

hip_l, shoulder_l, elbow_l, wrist_l, wrist_yaw_l = inverse_kinematics(*target_pos_l, desired_phi, yaw)
hip, shoulder, elbow, wrist, wrist_yaw = inverse_kinematics(*target_pos, desired_phi, yaw)

print("Angles (deg):", hip_l, shoulder_l, elbow_l, wrist_l, wrist_yaw_l)
print("Angles (deg):", hip, shoulder, elbow, wrist, wrist_yaw)

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1)  # Wait for Pico to reset
    print("Connected to Pico")

    # Send data to Pico
    # ser.write(b"Hello Pico!\n")
    data = struct.pack('Bhhhhhhhhhh', 3, hip_l, shoulder_l, elbow_l, wrist_l, wrist_yaw_l, hip, shoulder, elbow, wrist, wrist_yaw)

    print("balls")
    ser.write(data)
    while True:
        if ser.in_waiting:
            msg = ser.readline().decode('utf-8').strip()
            print(f"Pico \: {msg}")
        
except serial.SerialException as e:
    print(f"Error: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed.")

# 90 122 -150 -1 0