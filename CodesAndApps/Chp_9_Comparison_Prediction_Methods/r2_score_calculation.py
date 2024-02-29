# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:28:03 2024

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


from sklearn.metrics import r2_score
print("linear regresyon r2 value")
print(r2_score(Y, lin_reg.predict(X)))



# polinomial regression model

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

from sklearn.metrics import r2_score
print("polynomial regresyon r2 value")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


#linear regression predicting visualization
"""plt.scatter(X, Y)
plt.plot(x, lin_reg.predict(X))

#polinomial regression predicting visualization
plt.scatter(X, Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))

print(lin_reg.predict([[11]]))"""


# verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()

y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# Support Vector Regression
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')

svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli))
plt.show()

from sklearn.metrics import r2_score
print("SVR regresyon r2 value")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X, Y)

plt.scatter(X, Y)

plt.plot(X, r_dt.predict(X))
plt.show()


#  decision tree regresyon r2
from sklearn.metrics import r2_score
print("decision tree regresyon r2 value")
print(r2_score(Y, r_dt.predict(X)))

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0, n_estimators=10)

rf_reg.fit(X, Y.ravel())

#print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y)
plt.plot(X, rf_reg.predict(X))


from sklearn.metrics import r2_score
print("Random Forest r2 value")
print(r2_score(Y, rf_reg.predict(X)))






















