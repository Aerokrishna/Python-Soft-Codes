# axes[0] ==L_joy Left right
# axes[1] ==L_joy Up Down
# axes[2] ==L2 
# axes[3] ==R_joy Left right
# axes[4] ==R_joy Up Down
# axes[5] ==R2
# axes[6] ==L R arrows digital
# axes[7] ==U D arrows digital


# buttons[0] == x
# buttons[1] == o
# buttons[2] == △
# buttons[3] == ◻
# buttons[4] == L1
# buttons[5] == R1
# buttons[6] == L2 digital
# buttons[7] == R2 digital
# buttons[8] == Share
# buttons[9] == Options
# buttons[10] == Home
# buttons[11] == L_joy button
# buttons[12] == R_joy button
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist  
from nav_msgs.msg import Odometry
import math

class ps4SubscriberNode(Node):
    def __init__(self):
        super().__init__("joy_nav")
        self.vx=0
        self.vy=0
        self.vw=0

        self.vx_global=0
        self.vy_global=0
        self.I = [0,0,0]
        self.ps4_sub_= self.create_subscription(Joy ,"/joy", self.ps4_callback, 10)
        self.odom_sub_= self.create_subscription(Odometry ,"/odom", self.odom_callback, 10)
        self.ps4_pub_ = self.create_publisher(Twist, '/cmd_vel',10)
        
    def ps4_callback(self, msg: Joy):
        self.vx = 75*msg.axes[1]
        self.vy = 75*msg.axes[0]
        self.vw = 50*msg.axes[3]

        grab_prev = 0
        grab_now = msg.buttons[0]


        if (grab_prev == 0 and )
        
        # error_x = self.vx_target - msg_odom.twist.twist.linear.x
        # error_y = self.vy_target - msg_odom.twist.twist.linear.y
        # error_w = self.vw_target - msg_odom.twist.twist.angular.z 

        # self.I[0] = self.I[0] + error_x*0.1
        # self.I[1] = self.I[1] + error_y*0.1
        # self.I[2] = self.I[2] + error_w*0.1

        # #PID
        # vx = Kp[0] * error_x + Ki[0] * self.I[0]
        # vy = Kp[1] * error_y + Ki[1] * self.I[1]
        # vw = Kp[2] * error_w + Ki[2] * self.I[2]

    def odom_callback(self,odom_msg: Odometry):
         vel_msg = Twist()
         vel_msg.linear.x = (((self.vx)*math.cos(odom_msg.pose.pose.orientation.z))+((self.vy)*(math.sin(odom_msg.pose.pose.orientation.z))))
         vel_msg.linear.y = -(((self.vx)*math.sin(odom_msg.pose.pose.orientation.z))-((self.vy)*(math.cos(odom_msg.pose.pose.orientation.z))))
         vel_msg.angular.z = self.vw

         self.ps4_pub_.publish(vel_msg)
         print(vel_msg.linear.x, vel_msg.linear.y)
    
def main(args=None):
    rclpy.init(args=args)
    node = ps4SubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
