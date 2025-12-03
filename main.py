import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.regression.linear_model as sm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Separating independent variables and dependent variables
dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

# Applying OneHotEnconder
ct = ColumnTransformer(transformers=[('enconder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x = np.array(x, dtype=np.float64)

# Backward Elimination
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
x_opt = x[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
# print(regressor_OLS.summary())

# Separating Test and Train Data
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.3, random_state=0)

# Fit
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#  Inference
print(f'y_hat: {np.round(regressor.predict(x_test), 2)}; y: {y_test}')
 