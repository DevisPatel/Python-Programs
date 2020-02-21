from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

from sklearn.cluster import KMeans


data = pd.read_csv(r'C:\Users\patel\AppData\Local\Programs\Python\Python38\Programs\Assignment Programs\Day 10\Data Set\movie_metadata1.csv')
print (data.shape)
print (data.head)


print(data['director_facebook_likes'])
print(data.columns)

newdata=data.iloc[:,4:6]
print(newdata)

##############################    K Means Data and Cluster value assign   #################################################################

kmeans = KMeans(n_clusters=5)

kmeans.fit(newdata)

print(kmeans.cluster_centers_)
#print(len(kmeans.cluster_centers_))


print (kmeans.labels_)
print (len(kmeans.labels_))


print (type(kmeans.labels_))

unique, counts = np.unique(kmeans.labels_, return_counts=True)

print(dict(zip(unique, counts)))

# plot the data 

newdata['cluster'] = kmeans.labels_

sns.set_style('darkgrid')

sns.lmplot('director_facebook_likes', 'actor_3_facebook_likes',data=newdata, hue='cluster',palette='BuGn',size=6,aspect=1,fit_reg=False)







