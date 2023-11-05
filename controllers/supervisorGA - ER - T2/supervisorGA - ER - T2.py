from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import ga,os,sys,struct

class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.time_step = 32 # ms
        self.time_experiment = 75 # s
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters    
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")
        
        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Initialize the receiver and emitter data to null
        self.receivedData = "" 
        self.receivedWeights = "" 
        self.receivedFitness = "" 
        self.emitterData = ""
        
        ###########
        ### DEFINE here the 3 GA Parameters:
        self.num_generations = 30
        self.num_population = 2
        self.num_elite = 1
        
        # size of the genotype variable
        self.num_weights = 0
        
        # Creating the initial population
        self.population = []
        
        # All Genotypes
        self.genotypes = []
        
        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0
        self.prev_average_fitness = 0.0
        self.display.drawText("Fitness (Best - Red)", 0,0)
        self.display.drawText("Fitness (Average - Green)", 0,10)
        
        # Light
        self.light_node = self.supervisor.getFromDef("Light")
        if self.light_node is None:
            sys.stderr.write("No DEF Light node found in the current world file\n")
            sys.exit(1)
        self.light_on_field = self.light_node.getField("on")      

    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if(self.num_weights > 0):
            #  Size of the population and genotype
            pop_size = (self.num_population,self.num_weights)
            # Create the initial population with random weights
            self.population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            # Check Message 
            if(typeMessage == "weights"):
                self.receivedWeights = self.receivedData[9:len(self.receivedData)] 
                self.num_weights = int(self.receivedWeights)
            elif(typeMessage == "fitness"):  
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            self.receiver.nextPacket()
        
    def handle_emitter(self):
        if(self.num_weights > 0):
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            self.emitter.send(string_message)     
    
    def run_seconds(self,seconds):
        stop = int((seconds*1000)/self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if(stop == iterations):
                break    
            iterations = iterations + 1
    
    def reset_env(self, genotype, light_on_field_value):
            INITIAL_ROT = [-1.26771e-05, 1.18836e-05, 1, 1.63194]
            INITIAL_TRANS = [-0.685987, -0.66, -6.39627e-05]
            self.emitterData = str(genotype)
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()
            self.light_on_field.setSFBool(light_on_field_value)

    def calc_reward(self,AVOID_TRANS):
            FINAL_TRANS=np.array([0.105586,0.923258,0.00170258])
            FINAL_ROT=np.array([0.0,0.0,1,1.6])
            robot_trans=np.array(self.trans_field.getSFVec3f())
            robot_rot=np.array(self.rot_field.getSFVec3f())  
            delta_trans = 4/(abs(sum(robot_trans-FINAL_TRANS)))
            delta_rot = (abs(sum(robot_rot-FINAL_ROT)))*0.05
            robot_trans_avoid = -abs(sum(robot_trans - AVOID_TRANS)) * 0.008 

            print(round(delta_trans,2),round(robot_trans_avoid,2),round(delta_rot,2))
            reward=delta_trans+delta_rot+robot_trans_avoid
            return reward

    def evaluate_genotype(self,genotype,generation):
        # Here you can choose how many times the current individual will interact with both environments
        # At each interaction loop, one trial on each environment will be performed
        numberofInteractionLoops = 3
        currentInteraction = 0
        fitnessPerTrial = []
        AVOID_TRANS_LEFT=np.array([-0.23,0.3,0.5])
        AVOID_TRANS_RIGHT=np.array([0.35,0.26,0.05])
        while currentInteraction < numberofInteractionLoops:
            #######################################
            # TRIAL: TURN ?
            #######################################
            # Send genotype to robot for evaluation
            # Reset robot position and physics
            self.reset_env(genotype,False)
            # Evaluation genotype 
            # Measure fitness
            self.run_seconds(self.time_experiment)
            fitness = self.receivedFitness
            # Reward
            fitness+= self.calc_reward(AVOID_TRANS_LEFT)
            print("Fitness: {}".format(fitness))     
            # Add fitness value to the vector
            fitnessPerTrial.append(fitness)
            
            #######################################
            # TRIAL: TURN !?
            #######################################
            # Send genotype to robot for evaluation
            # Reset robot position and physics
            self.reset_env(genotype,True)
            # Evaluation genotype 
            # Measure fitness
            self.run_seconds(self.time_experiment)
            fitness = self.receivedFitness
            # Reward
            fitness+=self.calc_reward(AVOID_TRANS_RIGHT)
            print("Fitness: {}".format(fitness))

            # Add fitness value to the vector
            fitnessPerTrial.append(fitness)
            currentInteraction += 1
        fitness = np.mean(fitnessPerTrial)
        current = (generation,genotype,fitness)
        self.genotypes.append(current)  
        
        return fitness

    def run_optimization(self):
        # Wait until the number of weights is updated
        while(self.num_weights == 0):
            self.handle_receiver()
            self.createRandomPopulation()
        
        # For each Generation
        for generation in range(self.num_generations):
            print("Generation: {}".format(generation))
            current_population = []   
            # Select each Genotype or Individual
            for population in range(self.num_population):
                genotype = self.population[population]
                # Evaluate
                fitness = self.evaluate_genotype(genotype,generation)
                # Save its fitness value
                current_population.append((genotype,float(fitness)))
            # After checking the fitness value of all indivuals
            # Save genotype of the best individual
            best = ga.getBestGenotype(current_population)
            average = ga.getAverageGenotype(current_population)
            np.save("Best.npy",best[0])
            self.plot_fitness(generation, best[1], average)
            
            # Generate the new population using genetic operators
            if (generation < self.num_generations - 1):
                self.population = ga.population_reproduce(current_population,self.num_elite)
        
        #print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")   
    
    def draw_scaled_line(self, generation, y1, y2): 
        # the scale of the fitness plot
        XSCALE = int(self.width/self.num_generations)*5
        YSCALE = 500
        self.display.drawLine((generation-1)*XSCALE, self.height-int(y1*YSCALE), generation*XSCALE, self.height-int(y2*YSCALE))
    
    def plot_fitness(self, generation, best_fitness, average_fitness):
        if (generation > 0):
            self.display.setColor(0xff0000)  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness)
    
            self.display.setColor(0x00ff00)  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness)
    
        self.prev_best_fitness = best_fitness
        self.prev_average_fitness = average_fitness

    def run_demo(self):
        genotype = np.load("Best.npy")
        self.emitterData = str(genotype) 
        # Reset robot position and physics
        self.reset_env(genotype,False)
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment) 
        
        # Measure fitness
        fitness = self.receivedFitness
        print("Fitness without reward or penalty: {}".format(fitness))
        
        # Reset robot position and physics
        self.reset_env(genotype,False)
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment)  
        
        # Measure fitness
        fitness = self.receivedFitness
        print("Fitness without reward or penalty: {}".format(fitness))    
    
if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    gaModel = SupervisorGA()
    # Interface
    keyboard = Keyboard()
    keyboard.enable(50)
    print("***************************************************************************************************")
    print("To start the simulation please click anywhere in the SIMULATION WINDOW(3D Window) and press either:")
    print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
    print("***************************************************************************************************")
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if(resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
        elif(resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
        
        
