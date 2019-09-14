'''Importing the Libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

'''Importing the Dataset'''
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

'''Using the dendogram to find the optimal number of clusters'''
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))

'''Plot the Dendograms result'''
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

'''Applying Heirarchical Clustering to dataset'''
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
yc = hc.fit_predict(X)

'''Visualizing the cluster'''
plt.scatter(X[yc == 0, 0], X[yc == 0, 1], s = 100, color = 'red', label = 'Careful')
plt.scatter(X[yc == 1, 0], X[yc == 1, 1], s = 100, color = 'blue', label = 'Standard')
plt.scatter(X[yc == 2, 0], X[yc == 2, 1], s = 100, color = 'green', label = 'Target')
plt.scatter(X[yc == 3, 0], X[yc == 3, 1], s = 100, color = 'cyan', label = 'Careless')
plt.scatter(X[yc == 4, 0], X[yc == 4, 1], s = 100, color = 'magenta', label = 'Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
