import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

df = pd.read_csv("C:/Users/user/Desktop/DATASETS/Insurance Dataset.csv") #importing EastWest Airlines dataset
df.describe()

#  Creating Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalizing the data
df_norm = norm_func(df)

#scree plot or elbow curve 
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 2)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['cluster'] = clusters # creating a  new column and assigning it to new column 

df.head()
df_norm.head()

df = df.iloc[:,[5,0,1,2,3,4]]
df.head()

df.iloc[:, 1:].groupby(df.cluster).mean()

df.to_csv("insurance.csv", encoding = "utf-8")

import os
os.getcwd()
