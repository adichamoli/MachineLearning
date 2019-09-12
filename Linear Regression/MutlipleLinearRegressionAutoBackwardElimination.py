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

def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

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
SL = 0.05
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
print(X_Modeled.summary())