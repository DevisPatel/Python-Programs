import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

data = pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 14\Data Set\bio-degradabale-data.csv",sep=";", header=None)

data.columns = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42']

data.C42 = pd.Categorical(data.C42)
data['C42'] = data.C42.cat.codes

array = data.values

RB=[]
NRB=[]


for i in range(0,len(data)):

    if data['C42'][i]==1:

        RB.append(i)

    elif data['C42'][i]==0:

        NRB.append(i)


print("\n \n The Ready BioDegradable data in the given data set are                    :                         ",len(RB))

print("\n \n The Not Ready BioDegradable data in the given data set are                    :                         ",len(NRB))


random.seed(2)

pred_columns=data.iloc[:,0:41]

prediction_var=pred_columns.columns

print('\n \n ',list(prediction_var))



train, test = train_test_split(data, test_size = 0.3,random_state=46)

print("\n \n The shape of the Train Data is    :      ",train.shape)
print("\n \n The shape of the Test Data is    :      ",test.shape)


train_X = train[prediction_var]
train_y=train['C42']

test_X= test[prediction_var] 
test_y =test['C42']   


model = tree.DecisionTreeClassifier()
model.fit(train_X,train_y)

pred_y=model.predict(test_X)

print("\n \n The accuracy score is using DECISION TREE ALGORITHM is      :                                        ",metrics.accuracy_score(pred_y,test_y),'\n \n')


X = array[:,0:41]
Y = array[:,41]
kfold = model_selection.KFold(n_splits=10, random_state=8)
model = AdaBoostClassifier(n_estimators=30, random_state=8)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)

print("\n \n Accuracy after applying ADA BOOST ALGORITHM  is                          :                            ",results.mean(),'\n \n')



