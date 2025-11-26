import numpy as np
import matplotlib.pyplot as plt

# Link lengths
L1, L2, L3 = 12, 12, 9

def forward_kinematics(theta1, theta2, theta3):
    """Compute joint positions from angles"""
    # Shoulder
    x1, y1 = L1*np.cos(theta1), L1*np.sin(theta1)
    # Elbow
    x2, y2 = x1 + L2*np.cos(theta1+theta2), y1 + L2*np.sin(theta1+theta2)
    # Wrist / End effector
    x3, y3 = x2 + L3*np.cos(theta1+theta2+theta3), y2 + L3*np.sin(theta1+theta2+theta3)
    return np.array([[0,0],[x1,y1],[x2,y2],[x3,y3]])

def inverse_kinematics(x, y, z, phi, yaw):
    # hip rotation
    hip = np.atan2(y, x)

    r = np.sqrt(x**2 + y**2)
    print("r ", r)
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

    return hip, shoulder, elbow, wrist, yaw

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

    return hip, shoulder, elbow, wrist, yaw

def plot_arm(theta, target, phi, hip_angle):
    pts = forward_kinematics(*theta)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ---------- Left: planar arm ----------
    ax = axes[0]
    ax.plot(pts[:,0], pts[:,1], 'ko-', lw=2, label='Links')  # black links
    ax.scatter(*target, c='r', s=80, label='Target')
    ax.scatter(pts[-1,0], pts[-1,1], c='b', s=80, label='End effector')

    # draw orientation line of link3
    x_end, y_end = pts[-1]
    ax.plot([x_end, x_end+0.5*np.cos(phi)],
            [y_end, y_end+0.5*np.sin(phi)], 'g-', lw=2, label='Orientation')

    ax.set_title("Planar arm view (shoulder–elbow–wrist)")
    ax.axis('equal'); ax.grid(); ax.legend()

    # ---------- Right: top view showing hip rotation ----------
    ax2 = axes[1]
    ax2.plot([0, np.cos(hip_angle)], [0, np.sin(hip_angle)], 'k-', lw=3)
    ax2.scatter(0,0, c='b', s=60, label='Hip (base)')
    ax2.scatter(np.cos(hip_angle), np.sin(hip_angle), c='r', s=60, label='Direction')

    ax2.set_title("Top view (hip rotation)")
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()

world_frame = []
transform_W_base_left = []
transform_W_base_right = []


# Example usage
target_pos = (0, -20, 0)
r = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
desired_phi = np.deg2rad(-30)  # desired orientation = 45°
yaw = np.deg2rad(-30)  # desired orientation = 45°

hip, shoulder, elbow, wrist, wrist_yaw = inverse_kinematics(*target_pos, desired_phi, yaw)

print("Angles (deg):", np.degrees(hip), np.degrees(shoulder), np.degrees(elbow), np.degrees(wrist), np.degrees(yaw))
plot_arm((shoulder, elbow, wrist), (r, target_pos[2]), desired_phi, hip)

