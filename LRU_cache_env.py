# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:16:14 2020

@author: Arian Nsz
"""

import numpy as np
import pandas as pd
import gc

data = pd.read_csv('./data/requests.csv')

""" take a look at the shape"""

mem_status = np.zeros(shape=(9,6)) 
current_step = 0
mem_slots= [0, 3]
done = False
MAX_STEPS = 500
MIN_COMMUN= 5
MID_COMMUN= 2
MAX_COMMUN= -1
edge1 = np.array([0, 1, 2, 3, 4, 5])
edge2 = np.array([6, 7, 8, 9, 10, 11])
fresh_cost_weight= 1
reward=0
cache_count =0
lru_rank = np.array(range(18))
rewards=[]

while current_step<= MAX_STEPS:
    request = data.loc[current_step : current_step+1].values
    

    
    # update the table (considering life-time):
    # life = current_step - mem_status[:, [1,4]]
    # expire = life > mem_status[:, [2,5]]
    
    
    
    for i in range(2):
        edge1_has = np.where(mem_status[0:3, mem_slots] == request[i,0])
        edge2_has = np.where(mem_status[3:6, mem_slots] == request[i,0])
        parent_has = np.where(mem_status[6:, mem_slots] == request[i,0])
        do_cache = True
        
        if edge1_has[0].shape != (0,) and i==0:
            reward += MIN_COMMUN
            fresh= current_step - mem_status[edge1_has[0][0], (edge1_has[1][0]*3)+1]
            fresh /= mem_status[edge1_has[0][0], (edge1_has[1][0]*3)+2]
            if fresh>1:
                reward -= MAX_COMMUN
                mem_status[edge1_has[0][0], (edge1_has[1][0]*3)+2] =0
                mem_status[edge1_has[0][0], (edge1_has[1][0]*3)+1] =0
                mem_status[edge1_has[0][0], (edge1_has[1][0]*3)] =0
            else:
                reward -= fresh
                reward += MIN_COMMUN
        elif edge2_has[0].shape != (0,) and i==1:
            reward += MIN_COMMUN
            fresh= current_step - mem_status[edge2_has[0][0]+3, (edge2_has[1][0]*3)+1]
            fresh /= mem_status[edge2_has[0][0]+3, (edge2_has[1][0]*3)+2]
            if fresh>1:
                reward -= MAX_COMMUN
                mem_status[edge2_has[0][0]+3, (edge2_has[1][0]*3)+2] =0
                mem_status[edge2_has[0][0]+3, (edge2_has[1][0]*3)+1] =0
                mem_status[edge2_has[0][0]+3, (edge2_has[1][0]*3)] =0
            else:
                reward -= fresh
                reward += MIN_COMMUN
        elif parent_has[0].shape != (0,):
            reward += MID_COMMUN
            fresh= current_step - mem_status[parent_has[0][0]+6, (parent_has[1][0]*3)+1]
            fresh /= mem_status[parent_has[0][0]+6, (parent_has[1][0]*3)+2]
            if fresh>1:
                reward -= MAX_COMMUN
                mem_status[parent_has[0][0]+6, (parent_has[1][0]*3)+2] =0
                mem_status[parent_has[0][0]+6, (parent_has[1][0]*3)+1]=0
                mem_status[parent_has[0][0]+6, (parent_has[1][0]*3)]=0
            else:
                reward -= fresh
                reward += MIN_COMMUN
           
            """ The work in progress"""
    # LRU table update:
        if i==0:
            if parent_has[0].shape != (0,):
                do_cache= False
                # do the circular shift
                position= [parent_has[0][0], parent_has[1][0]*3]
                temp = 2*position[0] + int(position[1]/3)
                rank = np.where(lru_rank == temp)
                rank= rank[0][0] + 1
                lru_rank[:rank] = np.roll(lru_rank[:rank], 1)

            if edge1_has[0].shape != (0,):
                do_cache= False
                position= [edge1_has[0][0], edge1_has[1][0]*3]
                temp = 2*position[0] + int(position[1]/3)
                rank = np.where(lru_rank == temp)
                rank= rank[0][0] + 1
                lru_rank[:rank] = np.roll(lru_rank[:rank], 1)
                # do the circular shift
                # LRU table update:
            if do_cache:
                temp = lru_rank
                for num in edge2:
                    temp = np.delete(temp, np.where(temp == num))
                location = temp[-1]
                col = int((location % 2) *3)
                row = int((location - (location % 2)) / 2)
                mem_status[row, col] = request[i, 0]
                mem_status[row, col + 1] = current_step
                mem_status[row, col + 2] = request[i, 1]
                rank = np.where(lru_rank == location)
                rank = rank[0][0] + 1
                lru_rank[:rank] = np.roll(lru_rank[:rank], 1)

        if i==1:
            if parent_has[0].shape != (0,):
                do_cache= False
                # do the circular shift
                position= [parent_has[0][0], parent_has[1][0]*3]
                temp = 2*position[0] + int(position[1]/3)
                rank = np.where(lru_rank == temp)
                rank= rank[0][0] + 1
                lru_rank[:rank] = np.roll(lru_rank[:rank], 1)

            if edge2_has[0].shape != (0,):
                do_cache= False
                position= [edge2_has[0][0], edge2_has[1][0]*3]
                temp = 2*position[0] + int(position[1]/3)
                rank = np.where(lru_rank == temp)
                rank= rank[0][0] + 1
                lru_rank[:rank] = np.roll(lru_rank[:rank], 1)
        
            if do_cache:
                cache_count += 1
                temp = lru_rank
                for num in edge1:
                    temp = np.delete(temp, np.where(temp == num))
                location = temp[-1]
                col = int((location % 2) *3)
                row = int((location - (location % 2)) / 2)
                mem_status[row, col] = request[i, 0]
                mem_status[row, col + 1] = current_step
                mem_status[row, col + 2] = request[i, 1]
                rank = np.where(lru_rank==location)
                rank= rank[0][0] + 1
                lru_rank[:rank] = np.roll(lru_rank[:rank], 1)
    
    rewards.append(reward)
    print(f'loc: {location}')
    print(f'rank: {lru_rank}')
    print(f'Step: {current_step}')
    print(f'Next Requests: {data[current_step: current_step + 1].values}')
    print(f'Memory Status: {mem_status}')
    print(f'Reward: {reward}')
    current_step += 1

import matplotlib.pyplot as plt
plt.plot(rewards)