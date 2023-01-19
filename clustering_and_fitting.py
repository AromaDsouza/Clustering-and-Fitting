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
from sklearn import preprocessing

gdp_data=pd.read_csv('gdp growth.csv')
print(gdp_data) #data_countries

con_data=pd.read_csv('country continent.csv',encoding='latin')
con_data.sort_values(by='country',ascending=True,inplace=True)
print(con_data) #data_con

gdp_data=gdp_data.sort_values(by='Country Name',ascending=True)
data=gdp_data.merge(con_data,right_on='code_3',left_on='Country Code')
data=data.drop(['1993','1994','1995'],axis=1)
data=data.dropna(axis=0)
print(data)

#creating a column called average
data['average']=data.mean(numeric_only=True,axis=1)
data1=data.copy()
print(data)

dataframe=pd.DataFrame()
dataframe=data.copy()
data_new=pd.DataFrame()
data_new['countries']=data['Country Name']

label_encoder = preprocessing.LabelEncoder()
data['Country Name']= label_encoder.fit_transform(data['Country Name'])
data_cluster=pd.DataFrame()
data_cluster['country']=data['Country Name']
data_cluster['average']=data['average']

nclusters=3
kmeans=KMeans(n_clusters=nclusters)
kmeans.fit(data_cluster)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
print(cen)
print(skmet.silhouette_score(data_cluster, labels))

plt.figure(figsize=(10,10))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown","tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(nclusters): # loop over the different labels
    plt.plot(data_cluster[labels==l]["country"], data_cluster[labels==l]["average"],"o", markersize=3, color=col[l])

for ic in range(nclusters):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("Countries")
plt.ylabel("Average")
plt.show()

data_encoded_countries=pd.DataFrame()
data_encoded_countries['country names']=data_new['countries']
data_encoded_countries['country number']=data['Country Name']

dataframe.drop(['Country Code','continent','sub_region','average'],axis=1,inplace=True)

data_t=dataframe.T
header_row = 0
data_t.columns = data_t.iloc[header_row]
print(data_t)

data_t.columns = data_t.iloc[0]

data_t.drop('Country Name',axis=0,inplace=True)
data_t.reset_index(level=0,inplace=True)

data_t.rename(columns={'index':'year'},inplace=True)

dataframe_uk = pd.DataFrame()
dataframe_uk['year'] = data_t['year']
dataframe_uk['UK'] = data_t['United Kingdom']





