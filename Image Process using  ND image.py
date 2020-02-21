import matplotlib.pyplot as plt 
import numpy as np
from scipy import ndimage
from sklearn import cluster
from sklearn.cluster import KMeans
from PIL import Image


image = ndimage.imread("C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 10/Data Set/dogs.jpeg")
5
plt.figure(figsize = (15,8))

plt.imshow(image)


print("\n \n Current shape of the image is     :              ",image.shape)

x, y, z = image.shape
image_2d = image.reshape(x*y, z)

print("\n \n 2D shape of the image is      :               ",image_2d.shape)


kmeans_cluster = cluster.KMeans(n_clusters=7)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
print("\n \n The centers of the clusters are     :                \n \n ",cluster_centers)


unique, counts = np.unique(kmeans_cluster.labels_, return_counts=True)

print("\n Data occupied by the each cluster is            :                      ",dict(zip(unique, counts)))


cluster_labels = kmeans_cluster.labels_

print("\n \n The cluster labels are      :              ",cluster_labels)

plt.figure(figsize = (15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x,y*z))
plt.show()


