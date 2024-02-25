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
datas = pd.read_csv('../../DatasAndTools/veriler.csv')

print(datas)
    



# veri önişleme
boy = datas[['boy']]
print(boy)

