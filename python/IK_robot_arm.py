import numpy as np
import math
from sympy import *

#symbols
theta1 = symbols('theta1')
theta2 = symbols('theta2')
theta3 = symbols('theta3')
theta4 = symbols('theta4')
# p = symbols('p')
# q = symbols('q')
# r = symbols('r')
# s = symbols('s')

p = 1
q = 5
r = 5
s = 2

# Define the DH parameters
a = [0, q, r, s]
d = [p, 0, 0, 0]
alpha = [-math.pi/2, 0, 0, math.pi/2]

# Define the home position
theta_home = [0, 0, 0, 0]

# Define the goal position
x_goal = 8
y_goal = 0
z_goal = 0

yaw_goal = 0
pitch_goal = 0
roll_goal = 0

# define theta1 2 3 4
theta = [theta1,theta2,theta3,theta4]


def get_RotationMatrix(y,p,r):
    Rz = np.array([[np.cos(y),-np.sin(y),0],
                   [np.sin(y),np.cos(y),0],
                   [0,0,1]])
    
    Ry = np.array([[np.cos(p),0,-np.sin(p)],
                   [0,1,0],
                   [np.sin(p),0,np.cos(p)]])
    
    Rx = np.array([[1,0,0],
                   [0,np.cos(r),-np.sin(r)],
                   [0,np.sin(r),np.cos(r)]])
    
    
    R = Rz @ Ry @ Rx 

    return R

# Define the forward kinematics function
def forward_kinematics():
    T = [np.zeros((4, 4)) for _ in range(4)] #creates list of 4 tf matrices
    for i in range(4):
        T[i] = np.array([[ cos(theta[i]), ( -np.cos(alpha[i]) * sin(theta[i]) ), np.sin(alpha[i]) * sin(theta[i]), a[i] * cos(theta[i])],
                    [ sin(theta[i]), np.cos(alpha[i]) * cos(theta[i]),( -np.sin(alpha[i]) * cos(theta[i]) ), a[i] * sin(theta[i])],
                    [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                    [0, 0, 0, 1]
                    ])
       
        transform = T[0] @ T[1] @ T[2] @ T[3]
    return transform

def inverse_kinematics():
    tf = forward_kinematics()

    e1 = Eq(tf[0][3] - x_goal)
    e2 = Eq(tf[1][3] - y_goal)
    e3 = Eq(tf[2][3] - z_goal)

    R = get_RotationMatrix(yaw_goal,pitch_goal,roll_goal)

    e4 = Eq(tf[0][0] - R[0][0])
    e5 = Eq(tf[0][1] - R[0][1])
    e6 = Eq(tf[0][2] - R[0][2])
    e7 = Eq(tf[1][0] - R[1][0])
    e8 = Eq(tf[1][1] - R[1][1])
    e9 = Eq(tf[1][2] - R[1][2])
    e10 = Eq(tf[2][0] - R[2][0])
    e11 = Eq(tf[2][1] - R[2][1])
    e12 = Eq(tf[2][2] - R[2][2])


    sol = nsolve([e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12],[theta1,theta2,theta3,theta4],[np.pi/2,np.pi/6,np.pi/6,np.pi/10],tol=1e-3)
    for i in range(len(sol)):
        sol[i] = sol[i] * (180/np.pi)
    
    return sol

print(inverse_kinematics())



#try using quarternion