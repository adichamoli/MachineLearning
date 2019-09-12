'''
Equation
    y = b0 + b1*x1 + b2*(x1)^2 + ..... + bn*(x1)^n

where,
    y        = Dependent Variable
    x1,..,xn = Independent Variable
    b1,..,bn = Coeffecient
    b0       = Constant

Assumptions of Linear Regression
1. Linearity
2. Homoscedascity
3. Mutivariate normality
4. Independence of Errors
5. Lack of Multicollinearity
'''

'''Importing the Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

'''Importing the Dataset'''
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset.head())

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''Splitting Dataset into Training and Test set'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Since Data is small, so whole model is needed

'''Fitting Linear Regression to Dataset'''
lin_reg = LinearRegression()
lin_reg.fit(X, y)

'''Fitting Polynomial Regression to Dataset'''
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

'''Visualizing the Linear Regression Results'''
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''Visualizing the Polynomial Regression Results'''
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''Predicting a new result with Linear Regression'''
print(lin_reg.predict([[6.5]]))

'''Predicting a new result with Polynomial Regression'''
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))