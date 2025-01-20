import math
import sympy as sym

theta1 = 0
theta2 = 0

#leg_length = 0.107 = a
#foot_length = 0.130 = b
a = 0.107
b = 0.130

x = 0.1
y = 0.18
#self.initial_pose = [self.x,self.y]
#self.start_pose = [x,y]

l = math.sqrt((x**2) + (y**2))

phi = math.acos(( a**2 + l**2 - b**2)/(2 * a * l))

#shoulder angle
theta1 = phi + math.atan(y/x) - ((math.pi)/2)
# theta1 = math.degrees(theta1)
#knee angle
theta2 = math.pi - math.acos(( a**2 + b**2 - l**2)/(2 * a * b))
# theta2 = math.degrees(theta2)
if theta1<0:
    # print("theta1 = " ,theta1 + math.pi)
    print("theta1 = " ,math.degrees(theta1)+180)

print("theta2 = " ,math.degrees(theta2))
print("theta1 = " ,math.degrees(theta1))
print("theta1 = " ,theta1)
print("theta2 = " ,theta2)