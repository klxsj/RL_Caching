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
        self.mem_status= np.zeros(shape=(9,6))
        self.current_step = 0
        self.done = False
        self.MEM_SIZE = 6
        self.MAX_STEPS = 500
        self.MIN_COMMUN= 5
        self.MID_COMMUN= 2
        self.MAX_COMMUN= -1
        self.fresh_cost_weight= 1
        self.reward=0
        self.mem_slots= [0, 3]
        self.episodes=[]


        """
        Whether to cache a Content in one of the 6 memory slots of BS1
        Whether to cache a Content in one of the 6 memory slots of BS2
        Whether to cache a Content in one of the first 6 memory slots of the parent
        Whether to cache a Content in one of the second 6 memory slots of the parent
        """
        self.action_space = spaces.Box(low=0, high= 1,#self.MEM_SIZE
                                       shape=(8,), dtype= np.float16)
        self.observation_space= spaces.Box(
                                           low=0,
                                           high=40,
                                           shape= (9,6),
                                           dtype= np.uint8
                                          )

        """ observations are comprised of
            * 2 requested files (int)
            * The requested files' life-times (int)
            * memory status and the freshness (integrated)
                ** the files' id
                ** time of generation (integer)
                ** life time duration (integer)
                ** we are implicitly aware of the memory status
            * the BSs have the capacity to store 7 files
            * the parent node has the capacity of 14 files
        """
        
    def _next_observation(self):
        temp = self.df.loc[self.current_step: self.current_step+1].values
        self.mem_status[0,:] = np.array([temp[0,0], self.current_step,
                                         temp[0,1], temp[1,0],
                                         self.current_step, temp[1,1]])
        del temp
        gc.collect()
        obs = self.mem_status
        return obs
    
    def step(self, action):
        if self.current_step==0:
            self.episodes=[]
        """
        before taking any action we need to check if the requested files are
        available in the memory if requested files are not stored in memory then
        consider caching take and action return the *reward* and the new *observation*
        """
        request = self.df.loc[self.current_step : self.current_step+1 ].values
        action= (action*4)
        action =  np.round(action).astype(int)
        for i in range(2):
            if sum(action[i*4:4+4*i])!=0:
                self.reward+= self.MAX_COMMUN
            else:
                edge1_has = np.where(self.mem_status[1:4, self.mem_slots] == request[i,0])
                edge2_has = np.where(self.mem_status[4:7, self.mem_slots] == request[i,0])
                parent_has = np.where(self.mem_status[7:, self.mem_slots] == request[i,0])
                if edge1_has[0].shape != (0,) and i==0:
                    fresh= self.current_step - self.mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)+1]
                    # print('freshness is ' + str(fresh) )
                    # print('mem status is:')
                    # print(self.mem_status)
                    fresh /= self.mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)+2]
                    if fresh>1:
                        self.reward -= self.MAX_COMMUN
                        self.mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)+2] =0
                        self.mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)+1] =0
                        self.mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)] =0
                    else:
                        self.reward -= fresh
                        self.reward += self.MIN_COMMUN
                elif edge2_has[0].shape != (0,) and i==1:
                    fresh= self.current_step - self.mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)+1]
                    fresh /= self.mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)+2]
                    if fresh>1:
                        self.reward -= self.MAX_COMMUN
                        self.mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)+2] =0
                        self.mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)+1] =0
                        self.mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)] =0
                    else:
                        self.reward -= fresh
                        self.reward += self.MIN_COMMUN
                    
                elif parent_has[0].shape != (0,):
                    fresh= self.current_step - self.mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)+1]
                    fresh /= self.mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)+2]
                    if fresh>1:
                        self.reward -= self.MAX_COMMUN
                        self.mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)+2] =0
                        self.mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)+1] =0
                        self.mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)] =0
                    else:
                        self.reward -= fresh
                        self.reward += self.MID_COMMUN

        self._take_action(action)
        obs= self._next_observation()
        self.current_step += 1
        self.done = self.current_step > self.MAX_STEPS
        self.episodes.append(self.reward)
        return obs, -self.reward, self.done, {}


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
            