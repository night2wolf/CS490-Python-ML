# Tklinkenberg doing even number problems:
# 2)	Concatenate two dictionaries and sort the concatenated dictionary by value.
dictone = {'a':1,'b':2,'c':9,'d':4}
dicttwo = {'foo':7,'bar':5,'contoso':6}
mergeddict = {**dictone,**dicttwo}
print(mergeddict)
print(sorted(mergeddict.items(), key = lambda kv:(kv[1], kv[0]))) 
  
#4)	Go to https://catalog.umkc.edu/course-offerings/graduate/comp-sci/ 
# and fetch the course name and overview of course.
# Hint:Use BeautifulSoup package. 
import requests
from bs4 import BeautifulSoup
import urllib.request
import os
html = requests.get("https://catalog.umkc.edu/course-offerings/graduate/comp-sci/")
bsObj = BeautifulSoup(html.content,"html.parser")
names = []
descriptions = []
titles = []
p_Objs = bsObj.find_all("p")
for p_Obj in p_Objs:
   if p_Obj.attrs == {'class': ['courseblockdesc']}:
     descriptions.append(p_Obj.get_text())
class_names = bsObj.find_all("span")
for class_name in class_names:
  if class_name.attrs == {'class': ['code']}:
    names.append(class_name.get_text())
  if class_name.attrs == {'class': ['title']}:
    titles.append(class_name.get_text())
for i in range(len(names)):
  print(str(names[i]) + " " + str(titles[i]) + '\n' + str(descriptions[i]))

# 6)	Choose any dataset of your choice. Apply K-means on the dataset and visualize the clusters using matplotlib or seaborn.
# a. Report which K is the best using the elbow method.
# b. Evaluate with silhouette score or other scores relevant for unsupervised approaches
# (before applying clustering clean the data set with the EDA learned in the class)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
dataset = pd.read_csv('boston_house_prices.csv')
dataset = dataset.fillna(dataset.mean())
x = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
y = dataset.iloc[:,-1]

##elbow method to know the number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
# From elbow graph, 4 seems to be ideal clusters
# Calculate Silhouette score
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("Silhouette Score: ")
print(score)
# Silhouette Score: 
# 0.720652728235737
from sklearn.datasets.samples_generator import make_blobs
x,y_cluster_kmeans = make_blobs(n_samples=300, centers=3,cluster_std=0.60, random_state=0)
plt.scatter(x[:,0],x[:,1],c=y_cluster_kmeans,s=50,cmap='viridis')
plt.show()                       


# 8)	Create Multiple Regression by choosing a dataset of your choice
# (again before evaluating, clean the data set with the EDA learned in the class).
# Evaluate the model using RMSE and R2 and also report if you saw any improvement
# before and after the EDA.
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
data = datasets.load_boston(return_X_y=False)
boston = pd.read_csv('boston_house_prices.csv')
# print(data.DESCR)
##Build a linear model
y = np.log(boston.MEDV)
X = boston
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
predictions = model.predict(X_test)
print ("R^2 is: \n", r2_score(y_test, predictions))
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
# R^2 is: 0.9473926938136903
# RMSE is: 0.00844039597079636
##visualize
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()