#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

'''Splitting Dataset into Training and Test set'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Since Data is small, so whole model is needed

'''Fitting SVR to Dataset'''
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

'''Predicting a new result'''
y_pred = regressor.predict([[6.5]])
print(y_pred)

'''Visualizing the Decision Tree Regression Results'''
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
