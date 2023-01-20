# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:10:48 2023

@author: Aroma
"""
#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
from sklearn import preprocessing
import scipy.optimize as opt

#To read the first dataset (in csv format) and print it
gdp_data=pd.read_csv('gdp growth.csv')
#Sorting the values by Country Name in ascending order
gdp_data.sort_values(by='Country Name',ascending=True)  
print(gdp_data)

#To read the second dataset (in csv format) and print it
con_data=pd.read_csv('country continent.csv',encoding='latin')
#Sorting the values by country in ascending order
con_data.sort_values(by='country',ascending=True,inplace=True)
print(con_data)

#To merge the first and second datasets
data=gdp_data.merge(con_data,right_on='code_3',left_on='Country Code')
print(data)

#To drop and remove the unwanted columns
data=data.drop(['Indicator Name','Indicator Code','code_2','code_3','iso_3166_2',
                'country_code','region_code','sub_region_code','country', '1993',
                '1994','1995'],axis=1)

#To check if there are any missing (NAN) values
data.isnull().sum()

#To drop the rows with missing values
data=data.dropna(axis=0)

#To create a column called average
data['average']=data.mean(numeric_only=True,axis=1)
data1=data.copy()
print(data)

#To set the dataframe
dataframe=pd.DataFrame()
dataframe=data.copy()
data_new=pd.DataFrame()
data_new['countries']=data['Country Name']

#To encode labels in the column Country 
label_encoder = preprocessing.LabelEncoder()
data['Country Name']= label_encoder.fit_transform(data['Country Name'])
data

#To set up the dataframe to perform clustering with 2 columns country_name and average
data_cluster=pd.DataFrame()
data_cluster['country']=data['Country Name']
data_cluster['average']=data['average']
#To set up the number of clusters to perfom KMeans clustering
nclusters=3
kmeans=KMeans(n_clusters=nclusters)
#To fit the data
kmeans.fit(data_cluster)
labels = kmeans.labels_  #Labels are the number of associated clusters of (x,y)points
#To extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

#To calculate the silhoutte score
print(skmet.silhouette_score(data_cluster, labels))

#To plot the clusters, set the figure size and dpi is dots per inch i.e to set the resolution of the image and to produce a clear image
plt.figure(dpi=144, figsize=(10, 10))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown","tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(nclusters):  #Using loop over the different labels
    plt.plot(data_cluster[labels==l]["country"], data_cluster[labels==l]["average"],"*", markersize=3, color=col[l])
#To show the cluster centres
for inclusters in range(nclusters):
    xc, yc = cen[inclusters,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Countries")
plt.ylabel("Average")
plt.title("Clustering")
plt.savefig("plot1.png")  #To save the image
plt.show()  #To display the image of the bar graph

#To set up the encoded dataframe for country
data_encoded_countries=pd.DataFrame()
data_encoded_countries['country names']=data_new['countries']
data_encoded_countries['country number']=data['Country Name']
data_encoded_countries

#To fit and remove the unecessary columns
dataframe.drop(['Country Code','continent','sub_region','average'],axis=1,inplace=True)
data_transpose=dataframe.T  #To transpose the dataframe
#To convert the rows into header column
header_row = 0 
data_transpose.columns = data_transpose.iloc[header_row]
data_transpose.columns = data_transpose.iloc[0]  #To convert row to column header
data_transpose.drop('Country Name', axis=0, inplace=True)  #To drop the unwanted columns
data_transpose.reset_index(level=0, inplace=True)  #To reset the index values as column name
data_transpose.rename(columns={'index':'year'}, inplace=True)  #To rename the column

#To check fitting for the United Kingdom with respect to year
dataframe_uk=pd.DataFrame()  #To create dataframe for the UK
dataframe_uk['year']=data_transpose['year']
dataframe_uk['UK']=data_transpose['United Kingdom']
#To plot, set the figure size and dpi is dots per inch i.e to set the resolution of the image and to produce a clear image
plt.figure(dpi = 144, figsize=(20,20))
dataframe_uk.plot("year","UK")
plt.savefig("plot2.png")  #To save the image
plt.show()  #To display the image of the bar graph

#UK, USA and India are considered to be compared

def exp(t, n0, g):
    '''
    The above function named exp is to calculate the exponential function
    n0: scale factor
    g: growth rate
    '''
    t = t - 1990.0
    f = n0 * np.exp(g*t)
    return f

dataframe_uk["year"] = pd.to_numeric(dataframe_uk["year"])
param, covar = opt.curve_fit(exp, dataframe_uk["year"], dataframe_uk["UK"],p0=(73233967692.102798, 0.03))
dataframe_uk["Fit"] = exp(dataframe_uk["year"], *param)
dataframe_uk.plot("year", ["UK", "Fit"])
plt.savefig("plot3.png")  #To save the image
plt.show()  #To display the image of the bar graph

#To check fitting for the United States with respect to year
dataframe_usa=pd.DataFrame()  #To create a dataframe for the USA
dataframe_usa['year']=data_transpose['year']
dataframe_usa['Usa']=data_transpose['United States']
#To plot, set the figure size and dpi is dots per inch i.e to set the resolution of the image and to produce a clear image
plt.figure(dpi = 144, figsize=(20,20))
dataframe_usa.plot("year","Usa")
plt.savefig("plot4.png")  #To save the image
plt.show()  #To display the image of the bar graph

dataframe_usa["year"] = pd.to_numeric(dataframe_usa["year"])
param, covar = opt.curve_fit(exp, dataframe_usa["year"], dataframe_usa["Usa"],p0=(73233967692.102798, 0.03))

dataframe_usa["fit"] = exp(dataframe_usa["year"], *param)
dataframe_usa.plot("year", ["Usa", "fit"])
plt.savefig("plot5.png")  #To save the image
plt.show()  #To display the image of the bar graph

#To check fitting for the India with respect to year
dataframe_india=pd.DataFrame()  #To create a dataframe for India
dataframe_india['year']=data_transpose['year']
dataframe_india['india']=data_transpose['India']
#To plot, set the figure size and dpi is dots per inch i.e to set the resolution of the image and to produce a clear image
plt.figure(figsize=(20,20))
dataframe_india.plot("year","india")
plt.savefig("plot6.png")  #To save the image
plt.show()  #To display the image of the bar graph

dataframe_india["year"] = pd.to_numeric(dataframe_india["year"])
param, covar = opt.curve_fit(exp, dataframe_india["year"], dataframe_india["india"],p0=(73233967692.102798, 0.03))

dataframe_india["fit"] = exp(dataframe_india["year"], *param)
dataframe_india.plot("year", ["india", "fit"])
plt.savefig("plot7.png")  #To save the image
plt.show()  #To display the image of the bar graph

#Forecasting Future Years
#For the countries UK,USA,India
def log(t, n0, g, t0):
    '''
    The above function named log is to calculate the logistic function
    n0: scale factor
    g: growth rate
    t: time
    t0: estimated time
    '''
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

#Forecasting till 2030 for United Kingdom
#Non-linear least squares to fit a function to data.
param, covar = opt.curve_fit(log, dataframe_uk["year"], dataframe_uk["UK"],p0=(3e12, 0.03, 2000.0))
sigma = np.sqrt(np.diag(covar))  
print("parameters:", param)
print("std. dev.", sigma)
dataframe_uk["fit"] = log(dataframe_uk["year"], *param)
dataframe_uk.plot("year", ["UK", "fit"])
plt.savefig("plot8.png")  #To save the image
plt.show()  #To display the image of the bar graph

year = np.arange(1992, 2031)
forecast = log(year, *param)
plt.figure()
plt.plot(dataframe_uk["year"], dataframe_uk["UK"], label="UK")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("UK")
plt.legend()  #Legend shows the label names in form of a box on the upper right hand of the image
plt.savefig("plot9.png")  #To save the image
plt.show()  #To display the image of the bar graph

#Forecasting till 2030 for United States Of America
#Non-linear least squares to fit a function to data
param, covar = opt.curve_fit(log, dataframe_usa["year"], dataframe_usa["Usa"],p0=(3e12, 0.03, 2000.0),maxfev=5000)
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
dataframe_usa["fit"] = log(dataframe_usa["year"], *param)
dataframe_usa.plot("year", ["Usa", "fit"])
plt.savefig("plot10.png")  #To save the image
plt.show()  #To display the image of the bar graph

year = np.arange(1992, 2031)
forecast = log(year, *param)
plt.figure()
plt.plot(dataframe_usa["year"], dataframe_usa["Usa"], label="USA")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("Usa")
plt.legend()  #Legend shows the label names in form of a box on the upper right hand of the image
plt.savefig("plot11.png")  #To save the image
plt.show()  #To display the image of the bar graph

param, covar = opt.curve_fit(log, dataframe_india["year"], dataframe_india["india"],p0=(3e12, 0.03, 2000.0))
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
dataframe_india["fit"] = log(dataframe_india["year"], *param)
dataframe_india.plot("year", ["india", "fit"])
plt.savefig("plot12.png")  #To save the image
plt.show()  #To display the image of the bar graph

year = np.arange(1992, 2031)
forecast = log(year, *param)
plt.figure()
plt.plot(dataframe_india["year"], dataframe_india["india"], label="India")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("India")
plt.legend()  #Legend shows the label names in form of a box on the upper right hand of the image
plt.savefig("plot13.png")  #To save the image
plt.show()  #To display the image of the bar graph