# hip shoulder elbow wrist
# l1 l2 l3 l4
from sympy import *
import math as m

#arm specification
l1 = 0.33
l2 = 0.7
l3 = 0.7
l4 = 0.3

#goal pose specification
x = 1.3
y = 0
z = 0

orientation = m.pi/2

def inverse_kinematics():
    a = x - (l4 * m.cos(orientation))
    b = -(z - l1 + (l4 * m.sin(orientation)))
    # a = x  
    # b = z 
    theta1 = m.atan(y/x)
    theta3 = m.acos(( a ** 2 + b ** 2 - l2 ** 2 - l3 ** 2 )/( 2 * l2 * l3))
    theta2 = m.atan(b/a) - m.atan((l3 * m.sin(theta3))/(l2 + (l3 * m.cos(theta3))))
    theta4 =  theta2 + theta3 - orientation 

    # joint_angles = [(theta1), -(theta2), -(theta3),(theta4)]
    joint_angles = [(theta1), -(theta2), -(theta3),(theta4)]
    return joint_angles

print(inverse_kinematics())


