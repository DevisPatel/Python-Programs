from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

from sklearn.cluster import KMeans
from PIL import Image


im = Image.open("C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 10/Data Set/dogs.jpeg")

np_im = np.array(im)

print ("\n \n The image has been successfully tranlated into numpy array for better processing              :           ",type(np_im))

print("\n \n The current shape of the image is               :                 ",np_im.shape)

new_im = Image.fromarray(np_im)

new_im.save("C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 10/Data Set/image.png")

print("\n \n New  image has been saved.        -------->>>>>>>>>>            ",type(new_im))


x=np_im.shape[0]
y=np_im.shape[1]*np_im.shape[2]
z=np_im.shape[0]*np_im.shape[1]*np_im.shape[2]


np_im.resize((x,y))

print("\n \n The image has successfully converted into 2D  ARRAY        :              ", np_im)

print("\n \n The current shape of the image is               :                 ",np_im.shape)


kmeans = KMeans(n_clusters=3)

kmeans.fit(np_im)

cluster_centers = kmeans.cluster_centers_

print("\n \n The centeres of the clusters are              :               \n \n",cluster_centers)


unique, counts = np.unique(kmeans.labels_, return_counts=True)

print("\n Data occupied by the each cluster is            :                      ",dict(zip(unique, counts)))

cluster_labels = kmeans.labels_

print("\n \n The labels of the clusters are                 :                       ",cluster_labels)


plt.figure(figsize = (15,8))

plt.imshow(cluster_centers[cluster_labels])

plt.show()




