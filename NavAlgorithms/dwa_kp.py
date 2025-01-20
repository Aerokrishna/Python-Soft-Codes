import numpy as np
import matplotlib.pyplot as plt

# DWA PLANNER FOR A DIFFERENTIAL-DRIVE ROBOT

class Params():
    def __init__(self) -> None:
        self.min_vel = -0.5
        self.max_vel = 1.0
        self.min_w = - 40 * np.pi/180 # rad/s
        self.max_w =  40 * np.pi/180 # rad/s
        self.max_accel = 0.2
        self.time_step = 0.1
        self.time_period = 1.0
        self.v_resolution = 0.01
        self.w_resolution = 0.01

        self.speed_cost_gain = 1.0

def get_dynamic_window(state, params):

    # Dynamic window from robot specification
    Vs = [params.min_vel, params.max_vel,
          params.min_w, params.max_w]

    # Dynamic window from motion model
    Vd = [state[3] - params.max_accel * params.time_step,
          state[3] + params.max_accel * params.time_step,
          state[4] - params.max_w * params.time_step,
          state[4] + params.max_w * params.time_step] # getting the maximum and minimum values of final velocities achievable with given acceleration

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    
    # when given maximum decceleration to the current velocity, if the final velocity becomes less than the vmin of robot, limit it to v_min
    # when given maximum acceleration to current velocity, the final velocity should be within vmax

    return dw

def motion(state, control, dt):
    
    # calculate final state of the robot with respect to the given time and control command. Using differential drive model

    state[2] += control[1] * dt
    state[0] += control[0] * np.cos(state[2]) * dt
    state[1] += control[0] * np.sin(state[2]) * dt
    state[3] = control[0]
    state[4] = control[1]

    return state

def predict_trajectory(current_state, v, w, params):
    
    state = np.array(current_state)

    trajectory = np.array(state)
    time = 0

    while time < params.time_period:
        state = motion(state, [v, w], params.time_step)

        # stack the positions to form a trajectory
        trajectory = np.vstack((trajectory, state))
        time += params.time_step

    return trajectory

def get_best_trajectory(dw, state, params):
    current_state = state[:]
    min_cost = float("inf")
    best_control = [0.0, 0.0]
    best_trajectory = np.array([current_state])

    # evaluate all trajectory with sampled input in dynamic window
    traj_lis = []
    for v in np.arange(dw[0], dw[1], params.v_resolution):
        for w in np.arange(dw[2], dw[3], params.w_resolution):
            
            trajectory = predict_trajectory(current_state, v, w, params)
            traj_lis.append(trajectory)

            speed_cost = params.speed_cost_gain * (params.max_vel - trajectory[-1, 3])

            final_cost =  speed_cost

            if min_cost >= final_cost:
                min_cost = final_cost
                best_trajectory = trajectory
    
    return traj_lis, trajectory

params = Params()
      
# trajectory = predict_trajectory([0.0,0.0,0.0,1.0,0.0], 2.0, -0.1, params)

state = [0.0,0.0,0.0,0.3,0.0]
dw = get_dynamic_window(state, params)

trajs, best_traj = get_best_trajectory(dw, state, params)
# print(dw)
print(len(trajs))
for i in range(len(trajs)):

    plt.plot(trajs[i][:, 0], trajs[i][:, 1], "-r")

plt.plot(best_traj[:, 0], best_traj[:, 1], "-g")        
plt.show()
# print(get_best_trajectory())