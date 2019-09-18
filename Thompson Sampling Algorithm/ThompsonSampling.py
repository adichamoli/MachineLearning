'''Importing the Libraries'''
import random
import pandas as pd
import matplotlib.pyplot as plt

'''Import the Dataset'''
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
print(dataset.head())

'''Implementing Thompson Sampling'''
N = 10000
d = 10
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)

        if random_beta > max_random:
            max_random = random_beta
            ad = i

    ads_selected.append(ad)
    reward = dataset.values[n, ad]

    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1

    total_reward = total_reward + reward

print(total_reward)

'''Visualizing the Results'''
plt.hist(ads_selected)
plt.title('Histogram')
plt.xlabel('Ads')
plt.ylabel('No of times each ad was selected')
plt.show()