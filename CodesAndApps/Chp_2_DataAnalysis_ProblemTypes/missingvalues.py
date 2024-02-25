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
datas = pd.read_csv('../../DatasAndTools/eksikveriler.csv')

print(datas)
    



# veri önişleme
boy = datas[['boy']]
print(boy)
boykilo = datas[['boy','kilo']]
print(boykilo)


# eksik veriler 

# sci - kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

yas = datas.iloc[:,1:4].values
print(yas)

imputer.fit(yas[:,1:4])

yas[:,1:4] = imputer.transform(yas[:,1:4])

print(yas)




