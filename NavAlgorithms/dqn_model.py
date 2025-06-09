import torch.nn as nn

STATE_SIZE = 4   # x, y, goal_x, goal_y
ACTION_SIZE = 5  # forward, back, left, right, stop

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, state):
        return self.model(state)
