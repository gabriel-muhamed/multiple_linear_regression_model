import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

ct = ColumnTransformer(transformers=[('enconder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(f'y_hat: {np.round(regressor.predict(x_test), 2)}; y: {y_test}')
