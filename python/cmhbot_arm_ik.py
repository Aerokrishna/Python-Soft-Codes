import numpy as np
import matplotlib.pyplot as plt

# Link lengths
link1 = 5.21
link2 = 6.31

# Target goal position
goal_pose = [8.0, -1.5]  # Change this to any reachable point
x, y = goal_pose

# Compute inverse kinematics
distance = np.hypot(x, y)

# Check reachability
if distance > (link1 + link2):
    raise ValueError("Target is unreachable")

# Elbow angle (theta2)
cos_angle = (x**2 + y**2 - link1**2 - link2**2) / (2 * link1 * link2)
elbow_angle = -np.arccos(np.clip(cos_angle, -1.0, 1.0))  # clip for numerical safety

# Shoulder angle (theta1)
k1 = link1 + link2 * np.cos(elbow_angle)
k2 = link2 * np.sin(elbow_angle)
shoulder_angle = np.arctan2(y, x) - np.arctan2(k2, k1)

# Forward kinematics to get joint positions
joint_x = link1 * np.cos(shoulder_angle)
joint_y = link1 * np.sin(shoulder_angle)

end_eff_x = joint_x + link2 * np.cos(shoulder_angle + elbow_angle)
end_eff_y = joint_y + link2 * np.sin(shoulder_angle + elbow_angle)

# Print angles in degrees for reference
print("Shoulder angle (deg):", 90.0 + np.degrees(shoulder_angle))
print("Elbow angle (deg):", - np.degrees(elbow_angle))

# Plotting
plt.figure(figsize=(6, 6))
plt.plot([0, joint_x], [0, joint_y], 'r-', linewidth=3, label='Link 1')
plt.plot([joint_x, end_eff_x], [joint_y, end_eff_y], 'b-', linewidth=3, label='Link 2')
plt.plot(end_eff_x, end_eff_y, 'go', label='End Effector')
plt.plot(0, 0, 'ko', label='Base')

plt.xlim(-link1 - link2, link1 + link2)
plt.ylim(-link1 - link2, link1 + link2)
plt.gca().set_aspect('equal')
plt.grid(True)
plt.title('2DOF Arm Inverse Kinematics Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

