from controller import Robot
from datetime import datetime
import math
import numpy as np
MSPEED=10

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
        self.beacon=False
        self.inputs = []
        self.inputsPrevious = []
        self.flag_turn = 0
        self.ground = []
        self.proximity = []
        self.line_end = False

        # Enable Proximity Sensors
        self.proximity_sensors = []
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
    
    def follow_line(self):
        ground=self.ground
        print(ground,self.line_end )
        if not self.line_end:
            if ground[0]>450 and ground[1]>450 and ground[2]>450:
                self.lr(1,1)
            elif (ground[0]>450 and ground[1]>450 and ground[2]<450) or (ground[0]>450 and ground[1]<450 and ground[2]<450):
                self.lr(0.5,1)
            elif (ground[0]<450 and ground[1]>450 and ground[2]>450) or (ground[0]<450 and ground[1]<450 and ground[2]<450):
                self.lr(1,0.5)
            else:
                self.lr(0,0)
                self.line_end=True
        else:
            if self.beacon:
                self.lr(-1,1)
            else:
                self.lr(1,-1)
            if ground[0]<350 and ground[1]<350 and ground[2]<350:
                self.line_end=False
                self.lr(0,0)



    def sense_compute_and_actuate(self):
        self.follow_line()

    def run_robot(self):        
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # History
            self.groundp=self.ground
            self.proximityp=self.proximity
            self.ground = []
            self.proximity = []
            # Read Ground Sensors
            self.ground.append(self.right_ir.getValue())
            self.ground.append(self.center_ir.getValue())
            self.ground.append(self.left_ir.getValue())

            # Read Proximity Sensors
            for i in range(8):
                    temp = self.proximity_sensors[i].getValue()
                    # Adjust Values
                    min_ds = 0
                    max_ds = 2400
                    if(temp > max_ds): temp = max_ds
                    if(temp < min_ds): temp = min_ds
                    # Save Data
                    self.proximity.append((temp-min_ds)/(max_ds-min_ds))
            
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
    