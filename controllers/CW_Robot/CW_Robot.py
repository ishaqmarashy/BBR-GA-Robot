from controller import Robot
from datetime import datetime
import math
import numpy as np

class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        self.robot = robot
        self.time_step = 1 # ms
        self.max_speed = 5.3  # m/s
 
        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
        self.ground = []
        self.proximity = []
        self.proximity_sensors = []
        self.line_end = False
        self.stop = False
        self.go_around=False
        self.close=False
        self.beacon=False

        # Enable Proximity Sensors
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)
       
        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)
        
        # Enable Light Sensors
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)

    # helper function for managing motor speed
    def lr(self,l,r):
        self.velocity_left = l*self.max_speed
        self.velocity_right = r*self.max_speed
        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)  
          
    # state/event manager
    # sets enables/disables both avoid and line follow modules
    # signals when object is in proximity
    # signals enter/exit line event.
    def update(self):
        # stops we bot at "end state"
        if sum(self.proximity)>2:
            self.stop=True
        # check if object is in proximity
        if self.proximity[0] and self.proximity[7] :
            self.close=True
        elif not any(self.proximity[1:7]):
            self.close=False
        # line exit event
        if not self.go_around and (all(self.groundp) and not any(self.ground)) and self.close:
            self.go_around=True
        # line enter event
        elif self.go_around and (all(self.ground)):
            self.go_around=False
        # sets line to true when its detecting one
        self.line_end = any(self.ground)
        # enable go around if front sensors detect object
        # disable line follow module till next line enter event
        if any(self.proximity[0:2]) or any(self.proximity[6:8]):
            self.go_around=True

    # follow line module
    def follow_line(self):
        # left 0 center 1 right 2 
        if not self.go_around:
            if self.ground[0] and self.ground[1] and self.ground[2]:
                self.lr(0.7,0.7)
            elif self.ground[0] and self.ground[1] and not self.ground[2]:
                self.lr(0.2,0.7)        
            elif self.ground[0] and not self.ground[1] and not self.ground[2]:
                self.lr(0.2,0.7)
            elif not self.ground[0] and self.ground[1] and self.ground[2]:
                self.lr(0.7,0.2)
            elif not self.ground[0] and not self.ground[1] and self.ground[2]:
                self.lr(0.7,0.2)
            elif self.ground[0] and not self.ground[1] and self.ground[2]:
                self.lr(0.4,0.4)
            elif not self.beacon and not self.ground[0] and not self.ground[1] and not self.ground[2]:
                self.lr(0.5,0.1)
                pass
            elif not self.ground[0] and not self.ground[1] and not self.ground[2]:
                self.lr(0.1,0.5)
                pass
            elif not self.beacon:
                self.lr(0.1,0.5)
            else:
                self.lr(0.5,0.1)

    # object avoid module
    def avoid(self):
        # creates a turning preference based on "beacon signal"
        if not self.beacon:
            # front sensors are 0,1(right front) and 6,7 (left front)
            if any(self.proximity[0:2]) or any(self.proximity[6:8]):
                self.lr(0.8,-0.4)
            elif self.proximity[2]:
                self.lr(0.7,0.2)
            elif self.proximity[5]:
                self.lr(0.2,0.7)
            else:
                self.lr(0.1,0.5)
        else:
            if any(self.proximity[0:2]) or any(self.proximity[6:8]):
                self.lr(-0.4, 0.8)
            elif self.proximity[2]:
                self.lr(0.7,0.2)
            elif self.proximity[5]:
                self.lr(0.2,0.7)
            else:
                self.lr(0.5,0.1)

    def sense_compute_and_actuate(self):
        # run modules
        self.lr(0,0)
        self.update()
        self.avoid()
        self.follow_line()

    def run_robot(self):        
        while self.robot.step(self.time_step) != -1 and not self.stop:
            # history useful for exit and enter line events 
            self.groundp=self.ground
            self.proximityp=self.proximity
            # read and map ground sensors to True/False based on sensor value
            # 310 is a threshold for sensor values when it finds black colored floor 
            self.ground = [False if x > 310 else True for x in 
                           [self.left_ir.getValue(),
                            self.center_ir.getValue(),
                            self.right_ir.getValue()]]
            
            # read and map proximity sensor values to True/False 
            # 300 is a threshold for sensor values when it finds black colored floor 
            self.proximity = []
            for i in range(8):
                self.proximity.append(False if self.proximity_sensors[i].getValue() < 250 else True)
            
            # Check Light Sensors
            if not self.beacon:
                for i in range(8):
                        if self.light_sensors[i].getValue()<2000:
                            self.beacon=True
                            break
            # run modules
            self.sense_compute_and_actuate()
        # loop ended, print simulation time and stop motors 
        print("Time:",self.robot.getTime())
        self.lr(0,0)
      
if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
    