import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/user/Desktop/DATASETS/AutoInsurance.csv ")
df.describe() #gives iqr and std values.
df.info()
df1 = df.drop(['Customer','State','Effective To Date'],axis = 1) #not required as these variables dont require for calculation,removed 7th column as well(date)
df1.head() #displays first 5 observations
df1.columns # column names
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
#coverting categorical columns into numerical. 
#doing each column separately as each feature has diff values
df1['Response'] = le.fit_transform(df1['Response'])
df1['Coverage'] = le.fit_transform(df1['Coverage'])
df1['Education'] = le.fit_transform(df1['Education'])
df1['EmploymentStatus'] = le.fit_transform(df1['EmploymentStatus'])
df1['Gender'] = le.fit_transform(df1['Gender'])
df1['Location Code'] = le.fit_transform(df1['Location Code'])
df1['Marital Status'] = le.fit_transform(df1['Marital Status'])
df1['Policy Type'] = le.fit_transform(df1['Policy Type'])
df1['Policy'] = le.fit_transform(df1['Policy'])
df1['Renew Offer Type'] = le.fit_transform(df1['Renew Offer Type'])
df1['Sales Channel'] = le.fit_transform(df1['Sales Channel'])
df1['Vehicle Class'] = le.fit_transform(df1['Vehicle Class'])
df1['Vehicle Size'] = le.fit_transform(df1['Vehicle Size'])


#creating normalization function             
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df1)
df_norm.describe()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_norm, method='single',metric= 'euclidean')

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

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
clusters = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['cluster'] = clusters # creating a  new column and assigning it to new column 

df1.head()
df1 = df1.iloc[:,[21,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
df1.head()

final = df1.iloc[:,1:].groupby(df1.cluster).mean()
final
final.to_csv("telco_churn.csv", encoding = "utf-8")

import os
os.getcwd()
