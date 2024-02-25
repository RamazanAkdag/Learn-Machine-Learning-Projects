# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:58:08 2024

@author: akdag
"""
# ders 6 : kutuphanelerin yüklenmesi
            
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme
data = pd.read_csv('../../DatasAndTools/eksikveriler.csv')

print(data)
    



# veri önişleme
boy = data[['boy']]
print(boy)
boykilo = data[['boy','kilo']]
print(boykilo)


# eksik veriler 

# sci - kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

ulke = data.iloc[:,0:1].values
print(ulke)

#label encoding

"""ilk olarak ülke kolonu sayısal değere çevirilir ve sonra onu kategorik olacak şekilde encode etmemiz lazım"""
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke) # sayısal değere çevrildi

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)




