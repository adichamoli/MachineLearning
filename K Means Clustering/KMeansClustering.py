'''Importing the Libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

'''Importing the Dataset'''
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

'''Using the elbow method to find the optimal number of clusters'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

'''Plot the Elbow Methods result'''
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.show()

'''Applying K Means to dataset'''
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = kmeans.fit_predict(X)
print(y_means)

'''Visualizing the cluster'''
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, color = 'red', label = 'Careful')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, color = 'blue', label = 'Standard')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, color = 'green', label = 'Target')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, color = 'cyan', label = 'Careless')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, color = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, color = 'yellow', label = 'Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
