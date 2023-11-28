from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import os,sys,struct

class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        self.time_experiment = 90 # s

        # Please, do not change these parameters
        self.time_step = 32 # ms
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
        # Light
        self.light_node = self.supervisor.getFromDef("Light")
        if self.light_node is None:
            sys.stderr.write("No DEF Light node found in the current world file\n")
            sys.exit(1)
        self.light_on_field = self.light_node.getField("on")      
        #--------------------------------------------------------------------------------------

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            #Webots 2022: 
            #self.receivedData = self.receiver.getData().decode("utf-8")
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
            # Send genotype of an individual
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            #print("Supervisor send:", string_message)
            self.emitter.send(string_message)     
        
    def run_seconds(self,seconds):
        #print("Run Simulation")
        stop = int((seconds*1000)/self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if(stop == iterations):
                break    
            iterations = iterations + 1

    #-------------------------code between is our modification---------------------------
    def reset_env(self, left):
        self.boxr_r.setSFRotation(self.boxr_initial_rotation)
        self.obs_cyn1_t.setSFVec3f(self.obs_cyn1_initial_translation)
        self.boxl_t.setSFVec3f(self.boxl_initial_translation)
        self.boxl_r.setSFRotation(self.boxl_initial_rotation)
        self.obs_cyn2_t.setSFVec3f(self.obs_cyn2_initial_translation)
        INITIAL_ROT = [0.000585216, -0.000550635, 1, 1.63194]
        INITIAL_TRANS = [-0.685987, -0.66, -6.39627e-05]
        self.light_on_field.setSFBool(not left)
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        if left:
            self.emitterData = str('A')
        else:
            self.emitterData = str('B')

    def run_demo(self):
        self.reset_env(False)
        self.run_seconds(self.time_experiment) 
        
        self.reset_env(True)
        self.run_seconds(self.time_experiment)  
    

    
if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    gaModel = SupervisorGA()
    
    # Interface
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        gaModel.run_demo()
    #--------------------------------------------------------------------------------------

        
