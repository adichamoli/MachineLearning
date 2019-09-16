'''Importing libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori


'''Importing the Dataset'''
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
print(dataset.head())

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])

'''Training Apriori on the dataset'''
rules = apriori(transactions, min_support=0.03, min_confidence=0.2, min_lift=3, min_length=2)

'''Visualizing the results'''
results = list(rules)
print(results)