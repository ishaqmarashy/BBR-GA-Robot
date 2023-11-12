from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw
class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 3  # m/s
 
        self.number_input_layer = 12
        self.number_hidden_layer = [66,2]
        self.number_output_layer = 2
        
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []
        self.number_weights = 0
        for n in range(1,len(self.number_neuros_per_layer)):
            if(n == 1):
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
        self.velocity_left = 0
        self.velocity_right = 0
    
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

        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        self.receivedData = "" 
        self.receivedDataPrevious = "" 
        self.flagMessage = False
        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0

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
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]
        
        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left*self.max_speed)
        self.right_motor.setVelocity(self.velocity_right*self.max_speed)

    def calculate_fitness(self):

        lineFitness =3- (sum(self.inputs[0:3]))
        lineFitness *= 1   

        avoidCollisionFitness = 1-max(self.inputs[3:11])
        avoidCollisionFitness*= 1

        spinningFitness = 1-math.sqrt(abs(self.velocity_right - self.velocity_left) / (self.max_speed*2))
        spinningFitness *= 0.5

        forwardFitness = abs(self.velocity_right + self.velocity_left) / (self.max_speed*2)
        forwardFitness *= 2

        # print(
        #     round(lineFitness, 2),
        #     round(avoidCollisionFitness, 2),
        #     round(spinningFitness, 2),
        #     round(forwardFitness, 2)
        #     )        
        # print([round(value, 2) for value in self.inputs])
        ### DEFINE the fitness function equation of this iteration which should be a combination of the previous functions         
        combinedFitness = avoidCollisionFitness *lineFitness * forwardFitness *spinningFitness
        # print(round(combinedFitness, 2))

        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)


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
                # Adjust the Data to our model
                #Webots 2022:
                #self.receivedData = self.receiver.getData().decode("utf-8")
                #Webots 2023:
                self.receivedData = self.receiver.getString()
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                #print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()
                
            # Is it a new Genotype?
            if(np.array_equal(self.receivedDataPrevious,self.receivedData) == False):
                self.flagMessage = True
                # self.fitness = 0
            else:
                self.flagMessage = False
                
            self.receivedDataPrevious = self.receivedData 
        else:
            #print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):        
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []
            
            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()
            
            # Update your code without a separate function
            self.inputs += [x.getValue() /1023 for x in [self.left_ir, self.center_ir, self.right_ir]]

            # Read Distance Sensors
            self.inputs += [x.getValue() /4300  for x in self.proximity_sensors]

            # Beacon signal
            self.inputs += [sum([x.getValue()  for x in self.light_sensors])/(4300*8)]

     
            self.check_for_new_genes()
            self.sense_compute_and_actuate()
            self.calculate_fitness()

            
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
    
