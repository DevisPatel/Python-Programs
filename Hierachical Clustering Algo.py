import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('ggplot')

data = pd.read_csv(r'C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 10\Data Set\movie_metadata1.csv')

f1 = data['budget'].values
f2 = data['gross'].values
fb = f1[0:10]
fg = f2[0:10]

X = np.array(list(zip(fb, fg)))
Z = linkage(X, 'ward')

fig = plt.figure(figsize = (5,5))
dn = dendrogram(Z)

Z = linkage(X, 'single')

fig = plt.figure(figsize = (5,5))
dn = dendrogram(Z)

plt.show()



