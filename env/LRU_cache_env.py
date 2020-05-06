# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:16:14 2020

@author: arina
"""

import numpy as np
import pandas as pd
import gc

data = pd.read_csv('')

""" take a look at the shape"""

mem_status = np.zeros(shape=(8,6)) 
current_step = 0
mem_slots= [0, 3]
done = False
MAX_STEPS = 500
MIN_COMMUN= -1
MID_COMMUN= -2
MAX_COMMUN= -3
fresh_cost_weight= 1
reward=0

while current_step<= MAX_STEPS:
    request = data[current_step : current_step+1].values
    
    lru_rank = np.zeros(shape=(18,))
    
    # update the table (considering life-time):
    #life = current_step - mem_status[:, [1,4]]
    #expire = life > mem_status[:, [2,5]]
    
    
    
    for i in range(2):
        edge1_has = np.where(mem_status[1:4, mem_slots] == request[i,0])
        edge2_has = np.where(mem_status[4:7, mem_slots] == request[i,0])
        parent_has = np.where(mem_status[7:, mem_slots] == request[i,0])
        do_cache = True
        
        if edge1_has[0].shape != (0,) and i==0:
            reward += MIN_COMMUN
            fresh= current_step - mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)+1]
            fresh /= mem_status[edge1_has[0][0]+1, (edge1_has[1][0]*3)+2]
            reward -= fresh
            do_cache = False
        elif edge2_has[0].shape != (0,) and i==1:
            reward += MID_COMMUN
            fresh= current_step - mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)+1]
            fresh /= mem_status[edge2_has[0][0]+4, (edge2_has[1][0]*3)+2]
            reward -= fresh
            do_cache = False
        elif parent_has[0].shape != (0,):
            reward += MID_COMMUN
            fresh= current_step - mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)+1]
            fresh /= mem_status[parent_has[0][0]+7, (parent_has[1][0]*3)+2]
            reward -= fresh
            do_cache = False
            
    # LRU table update:
        if i==0:
            if parent_has[0].shape != (0,):
                # do the circular shift
                position= edge1_has[0][0], edge1_has[1][0]*3
                
            if edge1_has[0].shape != (0,):
                # do the circular shift
                    # LRU table update:
        if i==1:
            if parent_has[0].shape != (0,):
                # do the circular shift
                lru_rank[0:7] = np.roll(x[0:7],1)
            if edge2_has[0].shape != (0,):
                # do the circular shift
        
        if  do_cache:
            mem_status
            
            
                
    current_step += 1