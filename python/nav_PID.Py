import math
wI = 0
wKi =0
Kp = [2,2,1.8]

#set limits for BLACKBOARD CONTROLLER
acc_limit = 0.3
vel_limit = 1.5
w_limit = 2.0

class blackboard_controller():
    def __init__(self):
        self.prev_vx = 0
        self.prev_vy = 0
        
    def get_cmdVel(self,target,odom):
        global wI
        error_x = target[0] - odom[0]
        error_y = target[1] - odom[1]
        error_w = target[2] - odom[2]

        if math.pi<error_w<=2*math.pi:
            error_w = error_w - (2*math.pi)
        elif error_w<-math.pi:
            error_w = error_w + (2*math.pi)
        else:
            pass
                                                
        wI += error_w*0.1

        #PID
        vx = Kp[0] * error_x
        vy = Kp[1] * error_y
        vw = Kp[2] * error_w + wKi * wI

        if vx>0 and self.prev_vx>=0 and vx-self.prev_vx>acc_limit:
            vx=self.prev_vx+acc_limit 
        elif vx<0 and self.prev_vx<=0 and vx-self.prev_vx<-acc_limit :
            vx=self.prev_vx-acc_limit

        if vy>0 and self.prev_vy>=0 and vy-self.prev_vy>acc_limit :
            vy=self.prev_vy+acc_limit 
        elif vy<0 and self.prev_vy<=0 and vy-self.prev_vy<-acc_limit :
            vy=self.prev_vy-acc_limit

        vx = min(vx, vel_limit)
        vx = max(vx, -vel_limit)

        vy = min(vy, vel_limit)
        vy = max(vy, -vel_limit)

        vw = min(vw, w_limit)
        vw = max(vw, -w_limit)

        self.prev_vx = vx
        self.prev_vy = vy

        cmdVel_x = 75*(((vx)*math.cos(odom[2]))+((vy)*(math.sin(odom[2]))))
        cmdVel_y = -75*(((vx)*math.sin(odom[2]))-((vy)*(math.cos(odom[2]))))
        cmdVel_w = 50*vw     #analog values that needed to be put to cmd_vel
        
        return [cmdVel_x,cmdVel_y,cmdVel_w]

