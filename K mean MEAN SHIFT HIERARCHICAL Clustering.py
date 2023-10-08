#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
A=np.matrix([[1,2,3,4,11,10,12,23,56,7,88,99,12,6,7,78],
            [4,5,6,3,11,3,4,5,6,7,89,90,12,33,44,21],
            [7,8,9,6,33,44,23,45,67,65,87,78,99,90,12,18],
            [3,4,6,1,23,45,12,77,51,21,31,76,33,89,12,45],
            [6,7,8,9,54,11,23,73,92,12,55,66,87,43,52,23],
            [9,7,5,3,54,67,81,23,34,56,76,74,8,87,33,56]])
df= pd.DataFrame(A, columns = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16'])
df_std=(df-df.mean())/ (df.std())
n_components=3
from sklearn.decomposition import PCA
pca=PCA(n_components=n_components, svd_solver="randomized")
principalComponents= pca.fit_transform(df_std)
principalDf= pd.DataFrame(data=principalComponents,columns=['nf'+str(i+1) for i in range(n_components)])
print(principalDf)


# In[5]:


# Importing required libraries
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[6]:


# Step 1: Defining the number of clusters (K)
K = 3
# Step 2: Initializing cluster centroids
centroids = principalDf[:K]
centroids


# In[7]:


X=principalDf


# In[8]:


import numpy as np

def kmeans_clustering(X, K, centroids):
    while True:
        # Step 3: Assigning data points to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 4: Updating cluster centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # Checking convergence
        print("Old centroids --> ")
        print(centroids)
        print("New centroids --> ")
        print(new_centroids)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Running K-means clustering
K = 3  # Set the desired number of clusters
# Initialize centroids with appropriate shape (K, num_features)
centroids = np.random.rand(K, X.shape[1])
labels, centroids = kmeans_clustering(X.values, K, centroids)


# In[9]:


print(labels)


# In[14]:


import matplotlib.pyplot as plt

# Visualizing the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X['nf1'], X['nf2'], X['nf3'], c=labels)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='X', color='red', s=100)

ax.set_title("K-means Clustering")
ax.set_xlabel("nf1")
ax.set_ylabel("nf2")
ax.set_zlabel("nf3")
plt.show()


# In[18]:


import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Estimate the bandwidth with a reasonable quantile value (e.g., 0.2)
bandwidth = estimate_bandwidth(X.values, quantile=0.5)  # Adjust the quantile value as needed


# Perform Mean Shift clustering
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X.values)

# Get cluster labels and cluster centers
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Print cluster centers
print("Cluster Centers --> ")
print(cluster_centers)


# In[20]:


import matplotlib.pyplot as plt

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with nf1, nf2, and nf3
ax.scatter(X['nf1'], X['nf2'], X['nf3'], c=labels)

# Mark cluster centers in red
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='X', color='red', s=100)

ax.set_title("Mean Shift Clustering")
ax.set_xlabel("nf1")
ax.set_ylabel("nf2")
ax.set_zlabel("nf3")

plt.show()


# In[22]:


import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Create a sample dataset (replace this with your data)
# X = np.array([[1, 2], [1.5, 2.5], [3, 4], [6, 8], [8, 8], [10, 10]])

# Specify the number of clusters (you can also use linkage and distance_threshold parameters)
n_clusters = 3

# Perform agglomerative hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_clustering.fit(X)

# Get cluster labels
labels = agg_clustering.labels_

# Print cluster labels
print("Cluster Labels --> ")
print(labels)


# In[28]:


# Plot the clusters
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['nf1'], X['nf2'], X['nf3'], c=labels)  # Replace with your actual column names

# Set labels for each axis
ax.set_xlabel("nf1")
ax.set_ylabel("nf2")
ax.set_zlabel("nf3")

plt.title("Hierarchical Clustering")
plt.show()


# In[ ]:




