'''
- initial beleif states equal for all poses
- odometry model function (computes probability for given xt-1,ut,xt)
- compute bel*(xt)
- observation model function (computes probability for measured zt, and given xt)
- compute bel(xt)
'''

import numpy as np
import matplotlib.pyplot as plt


def gaussian_distribution(mean, std_dev=1, num_points=25):
    """
    Generate a Gaussian distribution with given mean and standard deviation.

    Parameters:
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.
    num_points (int): The number of points to generate. Default is 1000.

    Returns:
    x (numpy.ndarray): The x values.
    y (numpy.ndarray): The corresponding y values of the Gaussian distribution.
    """
    x = np.linspace(mean - 6*std_dev, mean + 6*std_dev, num_points)
    y = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(- (x - mean)**2 / (2 * std_dev**2))

    dis = {key: value for key, value in zip(x, y)}
    return dis

def normalize(belief):
    return belief / np.sum(belief)

def localization(zt,ut,states,beleif):
    # zt is the measurement we got

    prediction_beleif = [0.0] * states 
    for xt in range(states):

        dis = gaussian_distribution(xt)
        # motion update
        for xt_1 in range(states):
            s = xt_1 + ut
            if s in dis:
                prediction_beleif[xt] += (dis[s] * beleif[xt_1])
            else:
                prediction_beleif[xt] += 0
        
    # observation update
    for xt in range(states):
        dis = gaussian_distribution(xt)
        
        if zt in dis:
            beleif[xt] = 5 * dis[zt] * prediction_beleif[xt]
        else:
            beleif[xt] = 0
        
        #beleif[xt] = 5 * sensor(zt,xt) * prediction_beleif[xt]
        
    normalize(beleif)

    # print(prediction_beleif)
    return beleif

def plot_belief(belief):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(belief)), belief, color='blue', alpha=0.7)
    plt.ylim(0, 0.1)
    plt.xlabel('Position')
    plt.ylabel('Belief')
    plt.show()

states = 500
beleif = [0.1] * states
ut = 10

measurements = [37,49,57,63,70]
#plot_belief(localization(0,ut,states,beleif))
for zt in measurements:
    beleif = localization(zt,ut,states,beleif)
    print("state: ",beleif.index(max(beleif)))
    plot_belief(beleif)
    
    #print(localization(5,states,beleif))


