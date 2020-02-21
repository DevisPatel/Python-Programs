import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv('C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 8/Data Set/trans_us.csv', index_col = 0, thousands  = ',')

data.index.names = ['stations']
data.columns.names = ['months']
data = data.fillna(15)
data.head()



pca = PCA(n_components=2)
pca.fit(data)


existing_2d = pca.transform(data)
data_2d = pd.DataFrame(existing_2d)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']
data_2d.head()


print (pca.explained_variance_ratio_)
