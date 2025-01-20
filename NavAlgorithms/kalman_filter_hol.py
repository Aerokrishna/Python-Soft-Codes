import numpy as np
 
# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Extended Kalman Filter example (two-wheeled mobile robot)
 
# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)
 
# how state changes when no command executed
A_t_minus_one = np.array([[1.0,  0,   0],
                        [  0,1.0,   0],
                        [  0,  0, 1.0]])
 
# Noise applied to motion model
motion_model_noise = np.array([0.01,0.01,0.003])
     
# motion covarience noise (if decreased, we trust motion model more)
R_t = np.array([[1.0,   0,   0],
                [  0, 1.0,   0],
                [  0,   0, 1.0]])
                 
# To map the state xt to observation (changes if we get reading in some other form)
C_t = np.array([[1.0,  0,   0],
                [  0,1.0,   0],
                [  0,  0, 1.0]])
                         
# Sensor measurement noise covariance (if decreased, we trust observation model more)
Q_k = np.array([[1.0,   0,    0],
                [  0, 1.0,    0],
                [  0,    0, 1.0]])  

# Sensor noise
sensor_noise_w_k = np.array([0.07,0.07,0.04])
 
# describes how control ut changes the state from t-1 to t
def getB(yaw, deltak):
    """
    Calculates and returns the B matrix
    3x2 matix -> number of states x number of control inputs
    The control inputs are the forward speed and the
    rotation rate around the z axis from the x-axis in the 
    counterclockwise direction.
    [v,yaw_rate]
    Expresses how the state of the system [x,y,yaw] changes
    from k-1 to k due to the control commands (i.e. control input).
    :param yaw: The yaw angle (rotation angle around the z axis) in rad 
    :param deltak: The change in time from time step k-1 to k in sec
    """
    
    B = np.array([  [np.cos(yaw)*deltak, -np.sin(yaw)*deltak, 0],
                    [np.sin(yaw)*deltak, np.cos(yaw)*deltak, 0],
                    [0, 0, deltak]])
    return B
 
def ekf(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to 
    create an optimal estimate of the state of the robotic system.
         
    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            3x1 NumPy Array [v,v,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds
             
    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            3x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array                 
    """
    # Prediction step : A_t * x_t-1 + B_t * u_t + noise

    state_estimate_k = A_t_minus_one @ (
            state_estimate_k_minus_1) + (
            getB(state_estimate_k_minus_1[2],dk)) @ (
            control_vector_k_minus_1) + (
            motion_model_noise)
             
    print(f'Prediction={state_estimate_k}')
             
    # Predict the state covariance estimate based on the previous covarience and some noise (how spread is the prediction or how close is the prediction to real estimate)
    P_k = A_t_minus_one @ P_k_minus_1 @ A_t_minus_one.T + (
            R_t)
         
    # difference between the sensor measurement and the predicted state zt - C_t * predicted_state
    measurement_residual_y_k = z_k_observation_vector - (
            (C_t @ state_estimate_k) + (
            sensor_noise_w_k))
 
    print(f'Observation={z_k_observation_vector}')
             
    # predict the staet covarience estimate for the observation (will be used to calculate the weights)
    S_k = C_t @ P_k @ C_t.T + Q_k
         
    # Calculate the near-optimal Kalman gain
    K_k = P_k @ C_t.T @ np.linalg.pinv(S_k)
         
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
     
    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ C_t @ P_k)
     
    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={state_estimate_k}')
 
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
     
def main():
 
    # We start at time k=1
    k = 0
     
    # Time interval in seconds
    dk = 1
 
    # Create a list of sensor observations at successive timesteps
    # z_k = np.array([[4.721,0.143,0.006], # k=1
    #                 [9.353,0.284,0.007], # k=2
    #                 [14.773,0.422,0.009],# k=3
    #                 [18.246,0.555,0.011], # k=4
    #                 [22.609,0.715,0.012]])# k=5
    
    z_k = np.array([[2.123, 2.345, 0.0],
                    [4.343, 4.645, 0.007],
                    [7.123, 7.345, 0.009],
                    [8.823, 8.345, 0.012],
                    [10.123, 10.445, 0.017],
                    [12.123, 12.345, 0.022]])# k=5

    # Initial state estimate
    state_estimate_k_minus_1 = np.array([0.0,0.0,0.0])
     
    # control vectors
    control_vector_k_minus_1 = np.array([2.0,2.0, 0.0]) # x y w
     
    # State covariance matrix P_k_minus_1
    # This matrix has the same number of rows (and columns) as the 
    # number of states (i.e. 3x3 matrix). P is sometimes referred
    # to as Sigma in the literature. It represents an estimate of 
    # the accuracy of the state estimate at time k made using the
    # state transition matrix. We start off with guessed values.
    P_k_minus_1 = np.array([[0.1,  0,   0],
                            [  0,0.1,   0],
                            [  0,  0, 0.1]])
                             
    # Start at k=1 and go through each of the 5 sensor observations, 
    # one at a time. 
    # We stop right after timestep k=5 (i.e. the last sensor observation)
    for k, obs_vector_z_k in enumerate(z_k,start=1):
     
        # Print the current timestep
        print(f'Timestep k={k}')  

        # Run the Extended Kalman Filter and store the 
        # near-optimal state and covariance estimates
        optimal_state_estimate_k, covariance_estimate_k = ekf(
            obs_vector_z_k, # Most recent sensor measurement
            state_estimate_k_minus_1, # Our most recent estimate of the state
            control_vector_k_minus_1, # Our most recent control input
            P_k_minus_1, # Our most recent state covariance matrix
            dk) # Time interval
         
        # Get ready for the next timestep by updating the variable values
        state_estimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k
         
        # Print a blank line
        print()
 
# Program starts running here with the main method  
main()