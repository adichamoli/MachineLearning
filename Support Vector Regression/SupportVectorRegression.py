#Importing the Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

'''Splitting Dataset into Training and Test set'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Since Data is small, so whole model is needed

#Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))

'''Fitting SVR to Dataset'''
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

'''Predicting a new result'''
y_pred = regressor.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

'''Visualizing the Polynomial Regression Results'''
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()