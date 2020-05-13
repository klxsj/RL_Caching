import numpy as np
import pandas as pd
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt

numFiles= 40
x = np.arange(1, numFiles+1)
a = 1.1
weights = x ** (-a)
weights /= weights.sum()

#np.random.shuffle(weights)

bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))

lt= np.random.randint(4, 15, size=(numFiles,1))
files = bounded_zipf.rvs(size=500000)
lifetimes = lt[files-1]
requests = np.concatenate((files.reshape(-1,1), lifetimes.reshape(-1,1)), axis=1)
final_requests = pd.DataFrame(requests, columns=['Requested File', 'Lifetime'])
#plt.hist(requests[:,0], bins=np.arange(numFiles+2))
#plt.show()
del files
del requests
del lifetimes
gc.collect()
#plotting the histogram"


final_requests.to_csv('requests.csv')
