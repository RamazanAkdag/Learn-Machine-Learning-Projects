# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:10:15 2024

@author: akdag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data import
datas = pd.read_csv('../../DatasAndTools/maaslar.csv')

# data preprocessing

#data frame slicing
x=  datas.iloc[:,1:2]
y = datas.iloc[:,-1:]


#slices to numpy array
X = x.values
Y = y.values

# linear regression model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, Y)




# polinomial regression model

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#linear regression predicting visualization
plt.scatter(X, Y)
plt.plot(x, lin_reg.predict(X))

#polinomial regression predicting visualization
plt.scatter(X, Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))

print(lin_reg.predict([[11]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))