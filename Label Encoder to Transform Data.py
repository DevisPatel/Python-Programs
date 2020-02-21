import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm


data = pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 9\Data Set\voice-classification.csv")

print(data.head(6))


le = preprocessing.LabelEncoder()
data["label"] = le.fit_transform(data["label"])

print("\n \n The data set after tranforming the LABEL column is         \n",data['label'])


##########################    To Transform Every Column of the Data Set    ##########################################################

data[:]=preprocessing.MinMaxScaler().fit_transform(data)
data.head()

print("\n \n The data set after tranforming the whole Data Set is          \n",data)



'''

plt.subplots(4,5,figsize=(15,15))

for i in range(1,21):

    plt.subplot(4,5,i)
    plt.title(data.columns[i-1])
    sns.kdeplot(data.loc[data['label'] == 0, data.columns[i-1]], color= 'r', label='MALE')
    sns.kdeplot(data.loc[data['label'] == 1, data.columns[i-1]], color= 'k', label='FEMALE')

plt.show()


'''
    





