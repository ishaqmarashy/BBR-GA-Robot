from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import ga,os,sys,struct,math,csv

np.random.seed(0)  

class SupervisorGA:
    def __init__(self):
    #-------------------------code between is our modification---------------------------
        self.num_generations = 700
        self.num_population = 20
        self.num_elite = self.num_population*0.2
        
        # Simulation Parameters
        self.time_experiment = 100 # s
    # -----------------------------------------------------------------------------------

        # Please, do not change these parameters
        self.time_step = 32 # mss
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
        
    #-------------------------code between is our modification---------------------------
        self.boxr_node = self.supervisor.getFromDef("BOXR")
        self.boxr_t = self.boxr_node.getField("translation")
        self.boxr_r = self.boxr_node.getField("rotation")
        self.boxr_initial_translation = self.boxr_t.getSFVec3f()
        self.boxr_initial_rotation = self.boxr_r.getSFRotation()

        self.obs_cyn1_node = self.supervisor.getFromDef("OBS_Cyn1")
        self.obs_cyn1_t = self.obs_cyn1_node.getField("translation")
        self.obs_cyn1_r = self.obs_cyn1_node.getField("rotation")
        self.obs_cyn1_initial_translation = self.obs_cyn1_t.getSFVec3f()
        self.obs_cyn1_initial_rotation = self.obs_cyn1_r.getSFRotation()

        self.boxl_node = self.supervisor.getFromDef("BOXL")
        self.boxl_t = self.boxl_node.getField("translation")
        self.boxl_r = self.boxl_node.getField("rotation")
        self.boxl_initial_translation = self.boxl_t.getSFVec3f()
        self.boxl_initial_rotation = self.boxl_r.getSFRotation()

        self.obs_cyn2_node = self.supervisor.getFromDef("OBS_Cyn2")
        self.obs_cyn2_t = self.obs_cyn2_node.getField("translation")
        self.obs_cyn2_r = self.obs_cyn2_node.getField("rotation")
        self.obs_cyn2_initial_translation = self.obs_cyn2_t.getSFVec3f()
        self.obs_cyn2_initial_rotation = self.obs_cyn2_r.getSFRotation()
    # -----------------------------------------------------------------------------------
    
        ###########
        ### DEFINE here the 3 GA Parameters:
   
        # size of the genotype variable
        self.num_weights = 0
        
        # Creating the initial population
        self.population = []
        
        # All Genotypes
        self.genotypes = []
        
    #-------------------------code between is our modification---------------------------
        self.light_node = self.supervisor.getFromDef("Light")
        if self.light_node is None:
            sys.stderr.write("No DEF Light node found in the current world file\n")
            sys.exit(1)
        self.light_on_field = self.light_node.getField("on")      
    # -----------------------------------------------------------------------------------
    

    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if(self.num_weights > 0):
            #  Size of the population and genotype
            pop_size = (self.num_population,self.num_weights)
            # Create the initial population with random weights
            self.population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            #Webots 2023: 
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
            # Send genotype of an individual
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            #print("Supervisor send:", string_message)
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

    #-------------------------code between is our modification---------------------------
    # reward given to fitness if robot gets closer to pos
    def reward(self):
            FINAL_TRANS=np.array([0.10824,0.931462,0.00173902])
            robot_trans=np.array(self.trans_field.getSFVec3f())
            x=max(robot_trans[0],FINAL_TRANS[0])-min(robot_trans[0],FINAL_TRANS[0])
            y=max(robot_trans[1],FINAL_TRANS[1])-min(robot_trans[1],FINAL_TRANS[1])
            delta_trans = -math.sqrt(math.pow(x,2)+math.pow(y,2))*2
            reward=delta_trans
            print(f'G{delta_trans}')
            return reward

    def reset_env(self, genotype, left):
        self.boxr_t.setSFVec3f(self.boxr_initial_translation)
        self.boxr_r.setSFRotation(self.boxr_initial_rotation)
        self.obs_cyn1_t.setSFVec3f(self.obs_cyn1_initial_translation)
        self.boxl_t.setSFVec3f(self.boxl_initial_translation)
        self.boxl_r.setSFRotation(self.boxl_initial_rotation)
        self.obs_cyn2_t.setSFVec3f(self.obs_cyn2_initial_translation)
        INITIAL_ROT = [-1.26771e-05, 1.18836e-05, 1, 1.63194]
        INITIAL_TRANS = [-0.685987, -0.66, -6.39627e-05]
        self.emitterData = str(genotype)
        self.light_on_field.setSFBool(not left)
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        
    def evaluate_genotype(self,genotype,generation):
        fitnessPerTrial = []
        left=True
        # TRIAL: TURN RIGHT
        self.emitterData = str(genotype)
        self.reset_env(genotype,left)
        self.run_seconds(self.time_experiment)
        fitness = self.receivedFitness
        fitness+=self.reward()
        fitnessPerTrial.append(fitness)
        print("Fitness: {}".format(fitness))     

        # TRIAL: TURN LEFT
        self.emitterData = str(genotype)
        self.reset_env(genotype,not left)
        self.run_seconds(self.time_experiment)
        fitness = self.receivedFitness
        fitness+=self.reward()
        fitnessPerTrial.append(fitness)
        print("Fitness: {}".format(fitness))

        fitness = np.mean(fitnessPerTrial)
        current = (generation,genotype,fitness)
        self.genotypes.append(current)  
        
        return fitness

    def run_demo(self):
        left=True
        # Read File
        genotype = np.load("Best.npy")
        self.emitterData = str(genotype) 
        self.reset_env(genotype,left)
        self.run_seconds(self.time_experiment) 
        fitness = self.receivedFitness
        fitness+=self.reward()
        print("Fitness with reward : {}".format(fitness))
        
        # Turn Right
        self.emitterData = str(genotype) 
        self.reset_env(genotype,not left)
        self.run_seconds(self.time_experiment)  
        fitness = self.receivedFitness
        fitness+=self.reward()
        print("Fitness with reward: {}".format(fitness))    
    # -----------------------------------------------------------------------------------
    
    def run_optimization(self):
        #-------------------------code between is our modification---------------------------

        with open("fitness_data.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # Wait until the number of weights is updated
        # -----------------------------------------------------------------------------------
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
                    #print(fitness)
                    # Save its fitness value
                    current_population.append((genotype,float(fitness)))
                    #print(current_population)
                    writer.writerow([generation, fitness])                    
                # After checking the fitness value of all indivuals
                # Save genotype of the best individual
                best = ga.getBestGenotype(current_population);
                average = ga.getAverageGenotype(current_population);
                np.save("Best.npy",best[0])
                
                # Generate the new population using genetic operators
                if (generation < self.num_generations - 1):
                    self.population = ga.population_reproduce(current_population,self.num_elite);
                
            #print("All Genotypes: {}".format(self.genotypes))
            print("GA optimization terminated.\n")   
    
if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    gaModel = SupervisorGA()
    
    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)
    print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
    
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if(resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
            #print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        elif(resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
            #print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        
