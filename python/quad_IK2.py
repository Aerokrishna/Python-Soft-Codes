import math


theta1 = 0
theta2 = 0
theta3 = 0

#leg_length = 0.107 = b
#foot_length = 0.130 = c
a = 0.055
b = 0.107
c = 0.130

x = 0.1
z = 0.18
y = 0.1
#self.initial_pose = [self.x,self.z]
#self.start_pose = [x,z]

l = math.sqrt((x**2) + (z**2))

phi = math.acos(( b**2 + l**2 - c**2)/(2 * b * l))

alpha1 = math.atan((b+c)/a)
alpha2 = math.atan(y/z)

theta3 = alpha1 + alpha2 - math.pi/2
#shoulder angle
theta1 = phi + math.atan(z/x) - ((math.pi)/2)
# theta1 = math.degrees(theta1)
#knee angle
theta2 = math.pi - math.acos(( b**2 + c**2 - l**2)/(2 * b * c))
# theta2 = math.degrees(theta2)


if theta1<0:
    theta1 = math.pi + theta1

print("theta2 = " ,math.degrees(theta2))
print("theta1 = " ,math.degrees(theta1))
print("theta3 = ",math.degrees(theta3))
print("theta3 = ",theta3)
print("theta1 = " ,theta1)
print("theta2 = " ,theta2)