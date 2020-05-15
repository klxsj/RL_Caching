import numpy as np
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gc
import matplotlib.pyplot as plt
class cache_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, requests_df):
        super(cache_env, self).__init__()
        """   """
        self.df = requests_df
        self.mem_status= np.zeros(shape=(6,6))
        self.freshness= np.zeros(shape=(6,6))
        self.avg_fresh= 0
        self.current_step = 0
        self.done = False
        self.MEM_SIZE = 6
        self.MAX_STEPS = 500
        self.MIN_COMMUN= 5
        self.MID_COMMUN= 2
        self.MAX_COMMUN= -1
        self.fresh_cost_weight= 1
        self.reward=0
        self.mem_slots= [0, 2, 4]
        self.fresh_slots= [1, 3, 5]
        self.episodes=[]
        self.line1= np.zeros(shape=(1,6))


        """
        Defining Observation and Action Spaces
        """
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype= np.uint8)
        self.observation_space= spaces.Box(
                                           low=0,
                                           high=40,
                                           shape= (7,6),
                                           dtype= np.uint8
                                          )

        
    def _next_observation(self):
        temp = self.df.loc[self.current_step].values
        edge1_has = np.where(self.mem_status[0, :] == temp[0,0])
        edge2_has = np.where(self.mem_status[2, :] == temp[0,0])
        p_has = np.where(self.mem_status[4, :] ==  temp[0,0])

        if (edge1_has[0].shape != (0,)):
            edge1_has = 1
        if (edge2_has[0].shape != (0,)):
            edge2_has = 1
        if (p_has[0].shape != (0,)):
            p_has = 1 
            
        self.which_BS = self.current_step % 2
        self.line1 = np.array([temp[0,0], temp[0,1], edge1_has, 
                          edge2_has, p_has, self.which_BS]).reshape(1,6)
        obs = np.concatenate((self.line1, self.mem_status), axis=0)
        return obs
    
    def step(self, action):
        self.reward = 0
        if self.current_step==0:
            self.episodes=[]
        """
        before taking any action we need to check if the requested files are
        available in the memory. if requested files are not stored in memory then
        consider caching take and action return the *reward* and the new *observation*
        """
        request = self.df.loc[self.current_step].values
        edge1_has = np.where(self.mem_status[0, :] == request[0,0])
        edge2_has = np.where(self.mem_status[2, :] == request[0,0])
        parent_has = np.where(self.mem_status[4, :] == request[0,0])
        if (edge1_has[0].shape != (0,)):
            e1 = 1
        if (edge2_has[0].shape != (0,)):
            e2 = 1
        if (parent_has[0].shape != (0,)):
            p_has = 1
            
        for i in range(3):
            self.avg_fresh += (self.freshness[2*i+1,:] - self.freshness[2*i,:])/6
        
      #  calculate the reward for each case       
        if self.which_BS == 0:
            self.reward = e1 * self.MIN_COMMUN + p_has * self.MID_COMMUN
            if e1==1:
                self.reward -= self.mem_status[1, edge1_has[0][0]] #substract freshness cost
            elif p_has == 1:
                self.reward -= self.mem_status[5, parent_has[0][0]] #substract freshness cost
            else:
                self.reward += self.MAX_COMMUN()
                
            self.reward -= self.avg_fresh
            
        elif self.which_BS ==1:
            self.reward = e2 * self.MIN_COMMUN + p_has * self.MID_COMMUN
            if e2==1:
                self.reward -= self.mem_status[3, edge2_has[0][0]] #substract freshness cost
            elif p_has == 1:
                self.reward -= self.mem_status[5, parent_has[0][0]] #substract freshness cost
            else:
                self.reward += self.MAX_COMMUN()
    
            self.reward -= self.avg_fresh   
        
        expired = np.where(self.mem_status[self.fresh_slots, :] >= 1)
        self.mem_status[expired[0]*2, expired[1]] = 0 
        self.mem_status[expired[0]*2 + 1, expired[1]] = 0 
        

        self._take_action(action)
        obs= self._next_observation()
        self.current_step += 1
        self.done = self.current_step > self.MAX_STEPS
        self.episodes.append(self.reward)
        return obs, self.reward, self.done, {}


    def reset(self):
        self.mem_status= np.zeros(shape=(9,6))
        self.current_step = 0
        self.done = False
        self.reward= 0
        return self._next_observation()


    def _take_action(self, action):
        #print('action in _take_action() is:')
        #print(action)
        #action =  action.astype(int)#round(action).astype(int)
        request = self.df.loc[self.current_step : self.current_step+1 ].values
        for i in range(2):
            act = action[i*4:4+4*i]
            # print('action in for ' + str(i) + 'th request is: ')
            # print(act)
            for j in range(4):
                act0 = act[j]
                if act0 > 0:
                    row = int((act0-1)/2)
                    row += (2*j) +1
                    col= (act0-1)%2
                    col*= 3
                    self.mem_status[row, col] = request[i,0]
                    self.mem_status[row, col+1] = self.current_step
                    self.mem_status[row, col+2] = request[i,1]
                
            
            
    def render(self, mode='human', close= False):
        """   """
        print(f'Step: {self.current_step}')
        print(f'Next Requests: {self.df.loc[self.current_step: self.current_step+1].values}')
        print(f'Memory Status: {self.mem_status}')
        print(f'Reward: {self.reward}')
        if self.current_step==60 :
            plt.plot(self.episodes)
        print(len(self.episodes))
            