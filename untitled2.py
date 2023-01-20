# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:13:27 2023

@author: Aroma
"""

#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
# Import label encoder 
from sklearn import preprocessing

#importing the dataset
data_countries=pd.read_csv('gdp growth.csv')
print(data_countries)

#reading continent dataset
data_con=pd.read_csv('country continent.csv',encoding='latin')
data_con.sort_values(by='country',ascending=True,inplace=True)
print(data_con)

#considering country dataset
data_countries=data_countries.sort_values(by='Country Name',ascending=True)
print(data_countries)

#merging both country and continent datasets
data=data_countries.merge(data_con,right_on='code_3',left_on='Country Code')
print(data)

#removing year 1990
data=data.drop(['1993','1994','1995'],axis=1)
print(data)

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

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data['Country Name']= label_encoder.fit_transform(data['Country Name'])
print(data)

#setting up dataframe that requires columns need to be clustered
data_cluster=pd.DataFrame()
data_cluster['country']=data['Country Name']
data_cluster['average']=data['average']

# set up the clusterer for number of clusters
nclusters=3
kmeans=KMeans(n_clusters=nclusters)

# Fit the data, results are stored in the kmeans object
kmeans.fit(data_cluster) # fit done on x,y pairs

labels = kmeans.labels_
# print(labels) # labels is the number of the associated clusters of (x,y)␣,→points

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmet.silhouette_score(data_cluster, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown","tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(nclusters): # loop over the different labels
    plt.plot(data_cluster[labels==l]["country"], data_cluster[labels==l]["average"],"o", markersize=3, color=col[l])

# show cluster centres
for ic in range(nclusters):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("Countries")
plt.ylabel("Average")
plt.show()

#for understanding of encoded data of country
data_encoded_countries=pd.DataFrame()
data_encoded_countries['country names']=data_new['countries']
data_encoded_countries['country number']=data['Country Name']
print(data_encoded_countries)

#removing unecessary columns
dataframe.drop(['Country Code','continent','sub_region','average'],axis=1,inplace=True)
print(dataframe)

#transposing the dataframe for better use
data_t=dataframe.T

#converting rows into header column
header_row = 0
data_t.columns = data_t.iloc[header_row]
print(data_t)

# Convert row to column header using DataFrame.iloc[]
data_t.columns = data_t.iloc[0]
print(data_t)

#dropping unwanted columns
data_t.drop('Country Name',axis=0,inplace=True)
print(data_t)

#resetting index values as column name
data_t.reset_index(level=0,inplace=True)
print(data_t)

data_t.rename(columns={'index':'year'},inplace=True)
print(data_t)

#USA,UK,INDIA,CHINA countries are considered and compared
dataframe_uk = pd.DataFrame()
dataframe_uk['year'] = data_t['year']
dataframe_uk['UK'] = data_t['United Kingdom']

print(dataframe_uk['year'])
print(dataframe_uk['UK'])

#plotting the columns
plt.figure(figsize=(20,20))
dataframe_uk.plot("year","UK")
plt.show()

def exponential(t, n0, g):
#Calculates exponential function with scale factor n0 and growth rate g.
    t = t - 1990.0
    f = n0 * np.exp(g*t)
    return f
...


#Understanding of data
#Using Visualization libraries
#reading the data
data1
data1['continent'].unique()

def cont(x):
  data_a=data1[data1['continent']==x]
  fig=px.bar(y=data_a['Country Name'],x=data_a['average'],width=1000,height=1200)
  fig.update_layout(title_text=x, title_x=0.5)
  fig.show()

continents=['Asia','Europe','Africa','Americas','Oceania']
for i in continents:
  cont(i)

data_mostly=data1[data1['average']>5]
data_mostly
#using bar graph to plot countries that have more than 50% of land being covered
px.bar(y=data_mostly['average'],x=data_mostly['Country Name'],height=800,width=1000)
#px.update_layout(title_text='Bar Graph')


'''
#To create a dataframe for UK
dataframe_india=pd.DataFrame()
dataframe_india['year']=data_transpose['year']
dataframe_india['india']=data_transpose['India']
plt.figure(figsize=(20,20))
dataframe_india.plot("year","india")
plt.show()

dataframe_india["year"] = pd.to_numeric(dataframe_india["year"])
param, covar = opt.curve_fit(exp, dataframe_india["year"], dataframe_india["india"],p0=(73233967692.102798, 0.03))

dataframe_india["fit"] = exp(dataframe_india["year"], *param)
dataframe_india.plot("year", ["india", "fit"])
plt.show()

#To forecast for the future years for UK, USA and India
def logistic(t, n0, g, t0):
#"""Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

#To Forecast till 2030 for the United Kingdom
param, covar = opt.curve_fit(logistic, dataframe_uk["year"], dataframe_uk["UK"],p0=(3e12, 0.03, 2000.0))

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
dataframe_uk["fit"] = logistic(dataframe_uk["year"], *param)
dataframe_uk.plot("year", ["UK", "fit"])
plt.show()

year = np.arange(1992, 2031)
print(year)
forecast = logistic(year, *param)
plt.figure()
plt.plot(dataframe_uk["year"], dataframe_uk["UK"], label="UK")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("UK")
plt.legend()
plt.show()

#To Forecast till 2030 for United States Of America
param, covar = opt.curve_fit(logistic, dataframe_usa["year"], dataframe_usa["Usa"],p0=(3e12, 0.03, 2000.0),maxfev=5000)

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
dataframe_usa["fit"] = logistic(dataframe_usa["year"], *param)
dataframe_usa.plot("year", ["Usa", "fit"])
plt.show()

year = np.arange(1992, 2031)
print(year)
forecast = logistic(year, *param)

plt.figure()
plt.plot(dataframe_usa["year"], dataframe_usa["Usa"], label="Usa")
plt.plot(year, forecast, label="forecast")
plt.xlabel("Year")
plt.ylabel("Usa")
plt.legend()
plt.show()

param, covar = opt.curve_fit(logistic, dataframe_india["year"], dataframe_india["india"],p0=(3e12, 0.03, 2000.0))

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
dataframe_india["fit"] = logistic(dataframe_india["year"], *param)
dataframe_india.plot("year", ["india", "fit"])
plt.show()

year = np.arange(1992, 2031)
print(year)
forecast = logistic(year, *param)

plt.figure()
plt.plot(dataframe_india["year"], dataframe_india["india"], label="India")
plt.plot(year, forecast, label="forecast")
plt.xlabel("Year")
plt.ylabel("India")
plt.legend()
plt.show()
'''