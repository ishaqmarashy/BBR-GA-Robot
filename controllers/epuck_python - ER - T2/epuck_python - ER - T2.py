from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw
log=False
class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 5  # m/s
 
        self.number_input_layer = 12
        self.number_hidden_layer = [41,7,7]
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
        self.light_sensors.append(self.robot.getDevice('ls4'))
        self.light_sensors[0].enable(self.time_step)
        
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
        if log:
            self.forwardFitness = []
            self.forwardFitness_values = 0
            self.avoidCollisionFitness = []
            self.avoidCollisionFitness_values = 0
            self.spinningFitness = []
            self.spinningFitness_values = 0
            self.lineFitness = []
            self.lineFitness_values = 0
                    


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
        ### DEFINE the fitness function to increase the speed of the robot and 
        ### to encourage the robot to move forwardAVOID_TRANS_RIGHT
        
        forwardFitness = (8*(self.velocity_left + self.velocity_right)) / (2 * self.max_speed)
        forwardFitness*= 2.1
        ### DEFINE the fitness function equation to avoid collision
        avoidCollisionFitness = -sum(self.inputs[3:12]) / len(self.inputs[3:12]) 
        avoidCollisionFitness*= 1
        ### DEFINE the fitness function equation to avoid spining behaviour
        spinningFitness = min(self.velocity_right, self.velocity_left) - max(self.velocity_right, self.velocity_left)
        spinningFitness *= 0.5
        ### DEFINE the fitness function equation to promote line following 
        lineFitness = 1/(sum(self.inputs[0:3]) / len(self.inputs[0:3]))
        lineFitness*= 2.5
        

        ### DEFINE the fitness function equation of this iteration which should be a combination of the previous functions         
        combinedFitness = avoidCollisionFitness + spinningFitness + lineFitness + forwardFitness

        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)
        if log:
            self.spinningFitness.append(spinningFitness)
            self.spinningFitness_values = np.mean(self.spinningFitness) 
            self.forwardFitness.append(forwardFitness)
            self.forwardFitness_values = np.mean(self.forwardFitness) 
            self.avoidCollisionFitness.append(avoidCollisionFitness)
            self.avoidCollisionFitness_values = np.mean(self.avoidCollisionFitness) 
            self.lineFitness.append(lineFitness)
            self.lineFitness_values = np.mean(self.lineFitness) 

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send fitness:", string_message)

        if log:
            print('--')
            print(self.spinningFitness_values)
            print(self.forwardFitness_values)
            print(self.avoidCollisionFitness_values)
            print(self.lineFitness_values)

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
            
            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            min_gs = 0
            max_gs = 1023
            if(left > max_gs): left = max_gs
            if(center > max_gs): center = max_gs
            if(right > max_gs): right = max_gs
            if(left < min_gs): left = min_gs
            if(center < min_gs): center = min_gs
            if(right < min_gs): right = min_gs
            self.inputs.append((left)/(max_gs))
            self.inputs.append((center)/(max_gs))
            self.inputs.append((right)/(max_gs))

            
            # Read Distance Sensors
            for i in range(8):
                temp = self.proximity_sensors[i].getValue()
                min_ds = 0
                max_ds = 4300
                if(temp > max_ds): temp = max_ds
                if(temp < min_ds): temp = min_ds
                self.inputs.append((temp)/(max_ds))

            temp = self.light_sensors[0].getValue()
            min_ls = 0
            max_ls = 4300
            if(temp > max_ls): temp = max_ls
            if(temp < min_ls): temp = min_ls
            self.inputs.append((temp)/(max_ls))
            # GA Iteration       
            # Verify if there is a new genotype to be used that was sent from Supervisor  
            self.check_for_new_genes()
            # The robot's actuation (motor values) based on the output of the MLP 
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()
            # End of the iteration 

            
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
    
