import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_model import DQN

# Load the trained model
model = DQN()
model.load_state_dict(torch.load('dqn_model.pth'))
model.eval()

# Define discrete actions for a holonomic robot
ACTIONS = [
    np.array([0.1, 0.0]),   # Right
    np.array([-0.1, 0.0]),  # Left
    np.array([0.0, 0.1]),   # Up
    np.array([0.0, -0.1]),  # Down
    np.array([0.0, 0.0])    # Stay
]

# Robot + goal setup
robot_pos = np.array([0.0, 0.0])
goal_pos = np.array([1.0, 1.0])
trajectory = [robot_pos.copy()]

def get_state(pos, goal):
    return np.array([pos[0], pos[1], goal[0], goal[1]], dtype=np.float32)

# Move for max 100 steps or until close to goal
for _ in range(100):
    state = torch.FloatTensor(get_state(robot_pos, goal_pos)).unsqueeze(0)
    with torch.no_grad():
        q_vals = model(state)
    best_action = torch.argmax(q_vals).item()

    robot_pos += ACTIONS[best_action]
    trajectory.append(robot_pos.copy())

    if np.linalg.norm(robot_pos - goal_pos) < 0.1:
        print("Goal reached!")
        break

# Plot result
trajectory = np.array(trajectory)
plt.figure(figsize=(6,6))
plt.plot(trajectory[:,0], trajectory[:,1], marker='o', label='Trajectory')
plt.scatter([goal_pos[0]], [goal_pos[1]], c='green', label='Goal')
plt.scatter([trajectory[0,0]], [trajectory[0,1]], c='red', label='Start')
plt.grid(True)
plt.legend()
plt.title("Holonomic Robot Navigation with DQN")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.show()
