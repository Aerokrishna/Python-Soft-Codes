import numpy as np
import matplotlib.pyplot as plt

def initialize_belief(grid_size):
    # Initialize belief uniformly
    belief = np.ones(grid_size) / grid_size
    print(belief)
    return belief

def motion_model(belief, move_step, grid_size, p_correct=0.8):
    new_belief = np.zeros(grid_size)
    # edit all the states
    for i in range(grid_size):
        new_belief[i] = (p_correct * belief[(i - move_step) % grid_size] +
                         (1 - p_correct) * belief[i]) # calculate the posterior with motion update
    
    '''
    for every grid cell, update according to the motion model first.
    probability is employed according to the previous beleif state
    '''
    # 

    return new_belief

def sensor_model(belief, measurement, grid_size, landmarks, p_hit=0.6, p_miss=0.2):
    for i in range(grid_size):
        if i in landmarks and measurement == 1: # measurement 1 implies it senses the landmark
            belief[i] *= p_hit
        else:
            belief[i] *= p_miss
    return belief

def normalize(belief):
    return belief / np.sum(belief)

# Visualization function
def plot_belief(belief, step):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(belief)), belief, color='blue', alpha=0.7)
    plt.ylim(0, 1)
    plt.title(f'Belief Distribution at Step {step}')
    plt.xlabel('Position')
    plt.ylabel('Belief')
    plt.show()

# Parameters
grid_size = 10
move_step = 1
landmarks = [2, 4, 6, 9]  # Positions of landmarks
measurements = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # Example sensor measurements

# Initialize belief
belief = initialize_belief(grid_size)
print("Initial belief:", belief)
plot_belief(belief, 'Initial')

# Run Bayes filter with visualization
for step, measurement in enumerate(measurements, start=1):
    # Prediction step
    belief = motion_model(belief, move_step, grid_size)
    
    # Update step
    belief = sensor_model(belief, measurement, grid_size, landmarks)
    
    # Normalize the belief
    belief = normalize(belief)
    
    print(f"Updated belief after measurement {measurement} at step {step}:", belief)
    plot_belief(belief, step)
