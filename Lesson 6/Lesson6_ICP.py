from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics

dataset = pd.read_csv("CC.csv")
dataset = dataset.fillna(dataset.mean())
x = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]
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
# Calculate Silhouette score
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("Silhouette Score: ")
print(score)


#PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array,columns=x.columns)
pca = PCA(16)
x_pca = pca.fit_transform(x_scaled_array)
PCA_components = pd.DataFrame(x_pca)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

top_n_components = 2
x = PCA_components.iloc[:,:top_n_components]
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
# Recalculate K_means and silhouette score after PCA
nclusters = 4 
km = KMeans(n_clusters=nclusters)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x,y_cluster_kmeans)
print("new Silhoutte score after PCA: ")
print(score)

from sklearn.datasets.samples_generator import make_blobs
plt.scatter(x[0], x[1],c=y_cluster_kmeans, s=50,cmap='viridis')
centers = km.cluster_centers_
x, y_cluster_kmeans = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(centers[:, 0], centers[:, 1], s=200,alpha=.5)
plt.show()