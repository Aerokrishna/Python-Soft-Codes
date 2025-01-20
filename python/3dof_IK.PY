import numpy as np
import math
from sympy import *

#symbols
theta1 = symbols('theta1')
theta2 = symbols('theta2')
theta3 = symbols('theta3')
# p = symbols('p')
# q = symbols('q')
# r = symbols('r')
# s = symbols('s')

p = 1
q = 5
r = 5

# Define the DH parameters
a = [0, q, r]
d = [p, 0, 0]
alpha = [-np.pi/2, 0,np.pi/2]

# Define the home position
theta_home = [0, 0, 0]

# Define the goal position
x_goal = 5
y_goal = 4
z_goal = 0

# define theta1 2 3 4
theta = [theta1,theta2,theta3]


# Define the forward kinematics function
def forward_kinematics():
    T = [np.zeros((4, 4)) for _ in range(3)] #creates list of 4 tf matrices
    for i in range(3):
        T[i] = np.array([[ cos(theta[i]), ( -np.cos(alpha[i]) * sin(theta[i]) ), np.sin(alpha[i]) * sin(theta[i]), a[i] * cos(theta[i])],
                    [ sin(theta[i]), np.cos(alpha[i]) * cos(theta[i]),( -np.sin(alpha[i]) * cos(theta[i]) ), a[i] * sin(theta[i])],
                    [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                    [0, 0, 0, 1]
                    ])
       
        transform = T[0] @ T[1] @ T[2] 
    return transform

def inverse_kinematics():
    tf = forward_kinematics()

    e1 = Eq(tf[0][3] - x_goal)
    e2 = Eq(tf[1][3] - y_goal)
    e3 = Eq(tf[2][3] - z_goal)

    # -np.pi<theta1<np.pi
    # -np.pi<theta2<np.pi/12
    # -np.pi<theta3<np.pi

    sol = nsolve([e1,e2,e3],[theta1,theta2,theta3],[0.01,0.01,0.01],tol=1e-6)
    for i in range(len(sol)):
        sol[i] = sol[i] * (1/np.pi)
        
    return sol

print(inverse_kinematics())



