# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:10:48 2023

@author: Aroma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet

gdp_data=pd.read_csv('gdp growth.csv')
print(gdp_data) #data_countries

con_data=pd.read_csv('country continent.csv',encoding='latin')
con_data.sort_values(by='country',ascending=True,inplace=True)
print(con_data) #data_con

gdp_data=gdp_data.sort_values(by='Country Name',ascending=True)
print(gdp_data)

data=gdp_data.merge(con_data,right_on='code_3',left_on='Country Code')
print(data)

data=data.drop(['Indicator Name','Indicator Code','code_2','code_3','iso_3166_2','country_code','region_code','sub_region_code','country','1993','1994','1995'],axis=1)
print(data)

data=data.dropna(axis=0)
print(data)

data = data.isnull().sum()
print(data)

