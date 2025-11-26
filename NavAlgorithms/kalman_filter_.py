import numpy as np
import time
import matplotlib.pyplot as plt

class KalmanFilter():

    def __init__(self):

        # motion without any control commands (identity matrix)
        self.A = np.eye(3)
            
        # map state to observation (direct measurement of x,y,theta)
        self.C = np.eye(3)

        # fixed noise covariances
        self.R = 0.1 * np.eye(3)  # measurement noise
        self.Q = 0.1 * np.eye(3)  # process noise

    def getB(self, yaw, deltak, drive="holonomic"):
        if drive == "diff":
            B = np.array([[np.cos(yaw)*deltak, 0],
                          [np.sin(yaw)*deltak, 0],
                          [0, deltak]])
            return B
            
        elif drive == "holonomic":
            B = np.array([[np.cos(yaw)*deltak, -np.sin(yaw)*deltak, 0],
                          [np.sin(yaw)*deltak,  np.cos(yaw)*deltak, 0],
                          [0, 0, deltak]])
            return B
        else:
            raise ValueError("Unknown drive type")
                
    def predict(self, mean_, covariance_, control):
        mean_ = self.A @ mean_ + self.getB(mean_[2], 0.1) @ control
        covariance_ = self.A @ covariance_ @ self.A.T + self.Q
        return mean_, covariance_

    def compute_kalman_gain(self, covariance_):
        S_k = self.C @ covariance_ @ self.C.T + self.R
        K_gain = covariance_ @ self.C.T @ np.linalg.pinv(S_k)
        return K_gain

    def update(self, mean_, covariance_, reading):
        K_gain = self.compute_kalman_gain(covariance_)
        meas = reading - self.C @ mean_
        mean_ = mean_ + K_gain @ meas
        meas_cov = np.eye(3) - K_gain @ self.C
        covariance_ = meas_cov @ covariance_
        return mean_, covariance_

    def mock_sensor(self, mean, error=0.05):
        # add independent noise per dimension
        reading = mean + np.random.uniform(-error, error, size=mean.shape)
        return reading


def main():
    kf = KalmanFilter()
    control = np.array([0.1, 0.2, 0.0]) # constant control

    # initial pose and covariance
    mean_ = np.array([0.0, 0.0, 0.0])
    covariance_ = np.zeros((3,3))

    # store history for plotting
    times, preds, obs, fused = [], [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8,5))

    start = time.time()
    try:
        while True:
            # Kalman steps
            motion_mean, motion_cov = kf.predict(mean_, covariance_, control)
            observation = kf.mock_sensor(motion_mean)
            mean_, covariance_ = kf.update(motion_mean, motion_cov, observation)

            # append data
            t = time.time() - start
            times.append(t)
            preds.append(motion_mean.copy())
            obs.append(observation.copy())
            fused.append(mean_.copy())

            # live plot
            ax.clear()
            ax.plot(times, [p[0] for p in preds], 'b-', label="Predicted x")
            ax.plot(times, [o[0] for o in obs], 'r.', label="Observed x")
            ax.plot(times, [f[0] for f in fused], 'g-', label="Fused x")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("X Position")
            ax.legend()
            ax.set_title("Kalman Filter Live Plot")

            plt.pause(0.01)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped by user")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
