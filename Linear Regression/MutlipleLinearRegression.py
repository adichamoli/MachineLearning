'''
Equation
    y = b0 + b1*x1 + b2*x2 + ..... + bn*xn

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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''Importing the Dataset'''
dataset = pd.read_csv('Salary_Data.csv')
print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

'''Splitting Dataset into Training and Test set'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

'''Fitting Simple Linear Regression to Training Set'''
regressor = LinearRegression()
regressor.fit(X_train, y_train)

'''Predicting Test Set Results'''
y_pred = regressor.predict(X_test)
print(y_pred)

'''Visualizing the Traing Set Results'''
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Exp Training Set')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

'''Visualizing the Test Set Results'''
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Exp Test Set')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()