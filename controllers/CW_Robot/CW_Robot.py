from controller import Robot
from datetime import datetime
import math
import numpy as np

class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        self.robot = robot
        self.time_step = 1 # ms
        self.max_speed = 6.28  # m/s
 
        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
        self.go_around=False
        self.close=False
        self.beacon=False
        self.inputs = []
        self.inputsPrevious = []
        self.ground = []
        self.proximity = []
        self.proximity_sensors = []
        self.flag_turn = 0
        self.line_end = False
        self.end = False

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
    
    def lr(self,l,r):
        self.velocity_left = l*self.max_speed
        self.velocity_right = r*self.max_speed
        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)  
          
    def update(self):
        # stops we bot at "end state"
        if (all(self.proximity[0:2]) and all(self.proximity[6:8])):
            self.end=True
            print('STOP')
        # check if object is in proximity
        if self.proximity[0] and self.proximity[7] :
            self.close=True
        elif not any(self.proximity[1:7]):
            self.close=False
        # line exit
        if not self.go_around and (all(self.groundp) and not any(self.ground)):
            self.go_around=True
        # line enter
        elif self.go_around and (all(self.ground)):
            self.go_around=False
            self.beacon= not self.beacon

        # sets line to true when its detecting one
        self.line_end = any(self.ground)
        return self.line_end
        
    def follow_line(self):
        # print(self.groundint, "end?" ,self.line_end)
        # left center right 
        if self.ground[0] and self.ground[1] and self.ground[2]:
            self.lr(1,1)
        elif (self.ground[0] and self.ground[1] and not self.ground[2]) or\
        (self.ground[0] and not self.ground[1] and not self.ground[2]):
            self.lr(0.5,1)
        elif (not self.ground[0] and self.ground[1] and  self.ground[2]) or\
        (not self.ground[0] and not self.ground[1] and self.ground[2]):
            self.lr(1,0.5)
        elif not (self.ground[0] and self.ground[1] and self.ground[2]):
            if not self.beacon:
                print ('left')
                self.lr(-1,1)
            else:
                print('right')
                self.lr(1,-1)

    def avoid(self):
        self.proximity = self.proximity
        if self.proximity[0] or self.proximity[7]:
            self.lr(-1, 1)
            self.go_around=True
        elif self.proximity[2] or self.proximity[1]:
            self.lr(1,0.8)
        elif self.proximity[6] or self.proximity [5]:
            self.lr(0.8,1)

    def sense_compute_and_actuate(self):
        self.update()
        self.avoid()
        if not self.go_around:
            self.follow_line()
        if self.end:
            self.lr(0,0)

    def run_robot(self):        
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # History
            self.groundp=self.ground
            self.proximityp=self.proximity
            # Read Ground Sensors
            self.ground = [False if x > 450 else True for x in [self.left_ir.getValue(),self.center_ir.getValue(),self.right_ir.getValue()]]
            # Read Proximity Sensors
            self.proximity = []
            for i in range(8):
                self.proximity.append(False if self.proximity_sensors[i].getValue() < 300 else True)
            
            # Check Light Sensors
            if not self.beacon:
                for i in range(8):
                        temp = self.light_sensors[i].getValue()
                        # Adjust Values
                        min_ds = 0
                        max_ds = 2400
                        if(temp > max_ds): temp = max_ds
                        if(temp < min_ds): temp = min_ds
                        if temp<40:
                            self.beacon=True
            self.sense_compute_and_actuate()
      
                
            
if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
    