import numpy as np
import pandas as pd
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt

#%%
numFiles= 40
x = np.arange(1, numFiles+1)
a = 1.1
weights = x ** (-a)
weights /= weights.sum()


lt= np.random.randint(4, 15, size=(numFiles,1))
bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))
bs1 = bounded_zipf.rvs(size=150000)

plt.hist(bs1, bins=np.arange(numFiles+2))
plt.show()

#%%
x = np.arange(1, numFiles+1)
np.random.shuffle(x)
a = 1.1
weights = x ** (-a)
weights /= weights.sum()
np.random.shuffle(weights)
   
bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))
bs2 = bounded_zipf.rvs(size=150000)

plt.hist(bs2, bins=np.arange(numFiles+2))
plt.show()


#%%
x = np.arange(1, numFiles+1)
np.random.shuffle(x)
a = 1.1
weights = x ** (-a)
weights /= weights.sum()
np.random.shuffle(weights)
   
bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))
bs3 = bounded_zipf.rvs(size=150000)

plt.hist(bs3, bins=np.arange(numFiles+2))
plt.show()

#%%
  

files = []
for i in range(len(bs1 -1)) :
    files.append(bs1[i])
    files.append(bs2[i])
    files.append(bs3[i])
    
files = np.array(files)
    #%%
del bs1, bs2, bs3
#%%
lifetimes = lt[files-1]
requests = np.concatenate((files.reshape(-1,1), lifetimes.reshape(-1,1)), axis=1)
del files

final_requests = pd.DataFrame(requests, columns=['Requested File', 'Lifetime'])


del requests
del lifetimes
gc.collect()
#plotting the histogram"


final_requests.to_csv('requests.csv')
