import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque
from dqn_model import DQN  # <-- importing model class from external file

# === Parameters ===
STATE_SIZE = 4
ACTION_SIZE = 5
EPISODES = 10050
MAX_STEPS = 200
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Discrete Action Space ===
ACTIONS = [
    np.array([1.0, 0.0]),   # forward
    np.array([-1.0, 0.0]),  # backward
    np.array([0.0, 1.0]),   # left
    np.array([0.0, -1.0]),  # right
    np.array([0.0, 0.0])    # stop
]

# === Simple Holonomic Robot Env ===
class HolonomicEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = np.random.uniform(0, 1)
        self.y = np.random.uniform(0, 1)
        self.goal = np.array([0.9, 0.9])
        return self._get_state()

    def _get_state(self):
        return np.array([self.x, self.y, self.goal[0], self.goal[1]], dtype=np.float32)

    def step(self, action_index):
        vx, vy = ACTIONS[action_index]
        dt = 0.1
        self.x += vx * dt
        self.y += vy * dt

        done = False
        reward = -1  # small time penalty

        dist = np.linalg.norm(np.array([self.x, self.y]) - self.goal)
        if dist < 0.05:
            reward = 100
            done = True
        elif self.x < 0 or self.x > 1 or self.y < 0 or self.y > 1:
            reward = -100
            done = True

        return self._get_state(), reward, done

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

# === Training ===
def train():
    env = HolonomicEnv()
    q_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer()

    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            if len(buffer) > 64:
                s, a, r, s2, d = buffer.sample(64)

                s = torch.FloatTensor(s).to(DEVICE)
                a = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
                r = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
                s2 = torch.FloatTensor(s2).to(DEVICE)
                d = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)

                q_values = q_net(s).gather(1, a)
                next_q_values = target_net(s2).max(1)[0].detach().unsqueeze(1)
                targets = r + GAMMA * next_q_values * (1 - d)

                loss = torch.nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Sync target network
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    # === Save the trained model ===
    torch.save(q_net.state_dict(), 'dqn_model.pth')
    print("✅ Model saved as dqn_model.pth")

if __name__ == "__main__":
    train()
