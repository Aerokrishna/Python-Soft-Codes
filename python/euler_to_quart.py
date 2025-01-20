import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    # Convert angles from degrees to radians if necessary
    # roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return np.array([q_x, q_y, q_z, q_w])

# Example usage
roll, pitch, yaw = 0.0, 0.0, 0.17456

quaternion = euler_to_quaternion(roll, pitch, yaw)
print(quaternion)
