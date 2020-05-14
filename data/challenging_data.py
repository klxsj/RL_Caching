# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:41:47 2020

@author: arian92
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt

ones = np.ones(shape=(80000,))
others= np.random.randint(1,41, size=20000) 
bs1= np.concatenate((ones, others))
del ones, others
gc.collect()

#%%
random1 =[]
while len(random1)<8:
    temp = np.random.randint(1,41)
    if temp not in random1:
        random1.append(temp)
        
random =[]
while len(random)<8:
    temp = np.random.randint(1,41)
    if temp not in random:
        random.append(temp)
        
        
#%%
for i in range(len(random1)):
    bs1[i*10000 : i*10000+10000] = np.ones(shape=(10000)) * random1[i]
    
bs11 = bs1.astype(int)

#%%
ones = np.ones(shape=(80000,))
others= np.random.randint(1,41, size=20000) 
bs2= np.concatenate((ones, others))
del ones, others
gc.collect()
    
for i in range(len(random)):
    bs2[i*10000 : i*10000+10000] = np.ones(shape=(10000)) * random[i]

#%%
lt = np.ones(shape=(40,), dtype=int)

for i in range(len(random)):
    if i<= len(random)/2 - 1:
        lt[random[i]-1] = 2
        lt[random1[i]-1] = 2
    else:
        lt[random[i]-1] = 12
        lt[random1[i]-1] = 12

#%%
for x in range(len(lt)):
    if lt[x]==1:
        lt[x]= np.random.randint(3, 14)        
        
#%%
#file1 = np.array(bs1).astype(int)
lifetimes1 = lt[file1-1]

#%%
#file2 = np.array(bs2).astype(int)
lifetimes2 = lt[file2-1]
#%%

requests1= np.concatenate((file1.reshape(-1,1), lifetimes1.reshape(-1,1)), axis=1)
#%%
requests2= np.concatenate((file2.reshape(-1,1), lifetimes2.reshape(-1,1)), axis=1)
#%%
del file1, file2, lifetimes1, lifetimes2, lt, random, random1
gc.collect()

#%%    
final_requests= np.concatenate((requests1, requests2), axis=0)


#%%
final_requests=[]
np.random.shuffle(requests1)
np.random.shuffle(requests2)
for i in range(len(requests1)):
    final_requests.append(requests1[i,:])
    final_requests.append(requests2[i,:])
    
#%%
final_requests = pd.DataFrame(final_requests)
final_requests.columns= ['Requested File', 'Lifetime']
final_requests.to_csv('challenging_popularity.csv', index= False)




