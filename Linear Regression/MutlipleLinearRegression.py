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

Backward Elimination
    - Select a Significance level to stay in the model (e.g. SL = 0.05)
          |
    - Fit the full model with all possible predictors
          |
 |->- Consider the Predictor with the highets P-value. If P > SL then go to step 4, otherwise your model is ready
 |        |
 |  - Remove the predictor
 |        |
 ---- Fit the model without this variable

'''

'''Importing the Libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm

'''Importing the Dataset'''
dataset = pd.read_csv('50_Startups.csv')
print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

#Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

'''Avoiding Dummy Variable Trap'''
X = X[:, 1:]

'''Splitting Dataset into Training and Test set'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''Fitting Simple Linear Regression to Training Set'''
regressor = LinearRegression()
regressor.fit(X_train, y_train)

'''Predicting Test Set Results'''
y_pred = regressor.predict(X_test)
print(y_pred)

'''Building the optimal model using Backward Elimination'''
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

'''Repeating Step 3 of Backward Elimination to get P > 0.05'''
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())