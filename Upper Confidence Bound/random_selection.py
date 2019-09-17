'''Importing the Libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

'''Import the Dataset'''
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
print(dataset.head())

# Implementing Random Selection
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

print(total_reward)

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
#plt.show()