import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel("C:/Users/user/Desktop/DATASETS/Telco_customer_churn.xlsx ")
df.describe() #gives iqr and std values.
df.info()
df1 = df.drop(['Customer ID','Count','Quarter'],axis = 1)
df1.head()
df1.columns
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
#coverting categorical columns into numerical. 
#doing each column separately as each feature has diff values
df1['Referred a Friend'] = le.fit_transform(df1['Referred a Friend'])
df1['Offer'] = le.fit_transform(df1['Offer'])
df1['Phone Service'] = le.fit_transform(df1['Phone Service'])
df1['Multiple Lines'] = le.fit_transform(df1['Multiple Lines'])
df1['Internet Service'] = le.fit_transform(df1['Internet Service'])
df1['Internet Type'] = le.fit_transform(df1['Internet Type'])
df1['Online Security'] = le.fit_transform(df1['Online Security'])
df1['Online Backup'] = le.fit_transform(df1['Online Backup'])
df1['Device Protection Plan'] = le.fit_transform(df1['Device Protection Plan'])
df1['Premium Tech Support'] = le.fit_transform(df1['Premium Tech Support'])
df1['Streaming TV'] = le.fit_transform(df1['Streaming TV'])
df1['Streaming Movies'] = le.fit_transform(df1['Streaming Movies'])
df1['Streaming Music'] = le.fit_transform(df1['Streaming Music'])
df1['Unlimited Data'] = le.fit_transform(df1['Unlimited Data'])
df1['Contract'] = le.fit_transform(df1['Contract'])
df1['Paperless Billing'] = le.fit_transform(df1['Paperless Billing'])
df1['Payment Method'] = le.fit_transform(df1['Payment Method'])

#  Creating Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalizing the data
df_norm = norm_func(df1)

#scree plot or elbow curve 
TWSS = []
k = list(range(2, 10))
from sklearn.cluster import	KMeans
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['cluster'] = clusters # creating a  new column and assigning it to new column 

df.head()
df_norm.head()

df = df.iloc[:,[30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
df.head()

a = df1.iloc[:,1:].groupby(df.cluster).mean()
a
df.to_csv("telco_churn.csv", encoding = "utf-8")

import os
os.getcwd()
