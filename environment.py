import gym
from gym import spaces
import random
import numpy as np 

#environment should have the entire dataset as an input parameter, but train and test methods
class Environment(gym.Env):

    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = None
        self.state_space_dims = 256
        #self.action_space_dims = 1
        #actions are 0..15
        self.n_actions = 16

    def step(self, action, obs):
        #update state
        start_size = len(self.state)
        self.state += obs
        self.state = self.state[16:]
        next_state = self.state
        if (start_size != len(self.state)):
            print("Error in update state")

        reward = self.calculate_reward(action, obs)

        return next_state, reward
    
    def step5(self, action, obs,obs2,obs3,obs4,obs5):
        #update state
        start_size = len(self.state)
        self.state += obs
        self.state = self.state[16:]
        next_state = self.state
        if (start_size != len(self.state)):
            print("Error in update state")

        reward = self.calculate_reward5(action, obs,obs2,obs3,obs4,obs5)

        return next_state, reward

    #action takes values 0..15, so do indices of obs that has 16 values
    def calculate_reward5(self, action, obs,obs2,obs3,obs4,obs5):
        #print("obs-------------",obs)
        reward = 0.0
        x = obs[action]+obs2[action]+obs3[action]+obs4[action]+obs5[action]
        if ( x >= 3 ):
            reward = 1.0
        elif (x>=0 and x<=2 ):
            reward = -1.0
        else:
            print ("Error: channel quality should be 1 or 0")
        return reward
    
    def calculate_reward(self, action, obs):
        #print("obs-------------",obs)
        reward = 0.0
        if (obs[action] == 1):
            reward = 1.0
        elif (obs[action] == 0):
            reward = -1.0
        else:
            print ("Error: channel quality should be 1 or 0")
        return reward

    def reset(self, state_variables):
        self.state = state_variables
        return self.state

