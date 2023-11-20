from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw
np.random.seed(0)  

class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 4  # m/s
 
        # MLP Parameters and Variables 
        ###########
        ### DEFINE below the architecture of your MLP network:
        self.number_input_layer = 13
        self.number_hidden_layer = [13,13,13,13] 
        self.number_output_layer = 2
        
        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        
        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []
        
        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1,len(self.number_neuros_per_layer)):
            if(n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n-1]+1)*self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n]

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0.0
        self.velocity_right = 0.0
        self.ls_prev=0
        self.line_prev=0
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

        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)
        
        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        self.receivedData = "" 
        self.receivedDataPrevious = "" 
        self.flagMessage = False
        
        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0.0

    def check_for_new_genes(self):
        if(self.flagMessage == True):
                # Split the list based on the number of layers of your network
                part = []
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        part.append((self.number_neuros_per_layer[n-1]+1)*(self.number_neuros_per_layer[n]))
                    else:   
                        part.append(self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n])
                
                # Set the weights of the network
                data = []
                weightsPart = []
                sum = 0
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        weightsPart.append(self.receivedData[n-1:part[n-1]])
                    elif(n == (len(self.number_neuros_per_layer)-1)):
                        weightsPart.append(self.receivedData[sum:])
                    else:
                        weightsPart.append(self.receivedData[sum:sum+part[n-1]])
                    sum += part[n-1]
                for n in range(1,len(self.number_neuros_per_layer)):  
                    if(n == 1):
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1]+1,self.number_neuros_per_layer[n]])    
                    else:
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1],self.number_neuros_per_layer[n]])    
                    data.append(weightsPart[n-1])                
                self.network.weights = data
                
                #Reset fitness list
                self.fitness_values = []
        
    def sense_compute_and_actuate(self):
        # MLP: 
        #   Input == sensory data
        #   Output == motors commands
        # print(np.round(self.inputs,2))
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]
        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left*self.max_speed)
        self.right_motor.setVelocity(self.velocity_right*self.max_speed)

    def bin(self, min_val, max_val, val):
        if isinstance(val, list):
            return [self.bin(min_val, max_val, v) for v in val]
        else:
            if val > max_val:
                return 1.0
            elif val < min_val:
                return 0.0
            else:
                return (val-min_val)/(max_val-min_val)
            
    def calculate_fitness(self):
        ### DEFINE the fitness function to increase the speed of the robot and 
        ### to encourage the robot to move forward
        forwardFitness =self.bin(0,self.max_speed,(self.velocity_left+self.velocity_right))
        forwardFitness*= 1.0
        ### DEFINE the fitness function equation to line leaving behaviour
        lineFitness = self.inputs[12]
        lineFitness*= 1.0
        ### DEFINE the fitness function equation to avoid collision
        avoidCollisionFitness = 1-np.max(self.inputs[3:11])
        avoidCollisionFitness*= 1
        ### DEFINE the fitness function equation to avoid spining behaviour
        spinningFitness = 1-self.bin(0,self.max_speed,abs(self.velocity_left - self.velocity_right))
        spinningFitness*= 1.0
        if self.inputs[11]==1.0:
            turnFitness = 1.0 if self.velocity_right <= self.velocity_left else 0.4
        else:
            turnFitness = 1.0 if self.velocity_left <= self.velocity_right else 0.4

        ### DEFINE the fitness function equation of this iteration which should be a combination of the previous functions         
        combinedFitness = lineFitness*(spinningFitness+forwardFitness+turnFitness+avoidCollisionFitness)
        self.fitness_values.append(combinedFitness)
        fitm=np.mean(self.fitness_values) 
        # print(np.round([spinningFitness,forwardFitness, lineFitness,turnFitness,fitm],2))
        # print(np.round(self.inputs[11:],2))
        # print(np.round(self.inputs,2))
        # print(round(combinedFitness,3))
        # print(self.inputs[11])
        self.fitness = fitm

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)
            
    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while(self.receiver.getQueueLength() > 0):
                self.receivedData = self.receiver.getString()
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                self.receiver.nextPacket()
            if(np.array_equal(self.receivedDataPrevious,self.receivedData) == False):
                self.flagMessage = True
                self.ls_prev=0
                self.fitness = 0.0
            else:
                self.flagMessage = False
            self.receivedDataPrevious = self.receivedData 
        else:
            self.flagMessage = False

    def run_robot(self):        
        self.inputs =[]
        while self.robot.step(self.time_step) != -1:
            self.inputs =[]
            self.handle_emitter()
            self.handle_receiver()
            
            self.inputs+=self.bin(350,500,[self.left_ir.getValue(),self.center_ir.getValue(),self.right_ir.getValue()])
            self.inputs+=self.bin(100,200,[x.getValue() for x in self.proximity_sensors])

            # while above 0.3 input will be 1 and when its below input will be 0
            ls=self.bin(300,500,min([x.getValue()  for x in self.light_sensors]))
            if ls==0:
                self.ls_prev=ls
            else:
                # 0.155 at 60s
                self.ls_prev=self.ls_prev*0.999

            if 1-np.min(self.inputs[0:3])==1.0:
                self.line_prev=1
            else:
                # 0.04 at 10s
                self.line_prev*=0.99

            self.inputs+=[1.0 if self.ls_prev > 0.155 else 0]
            self.inputs+=[1.0 if self.line_prev > 0.04 else 0]
            self.inputs=np.round(self.inputs,3)

            self.check_for_new_genes()
            self.calculate_fitness()
            self.sense_compute_and_actuate()
            
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
    
