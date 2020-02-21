import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv(r"C:\Users\patel\AppData\Local\Programs\Python\Python38\Programs\Assignment Programs\Day 6\Data Set\pacific.csv")

print(data.head(6))


data.Status = pd.Categorical(data.Status)
data['Status'] = data.Status.cat.codes


random.seed(2)
pred_columns = data[:]
pred_columns.drop(['Status'],axis=1,inplace=True)
pred_columns.drop(['Event'],axis=1,inplace=True)
pred_columns.drop(['Latitude'],axis=1,inplace=True)
pred_columns.drop(['Longitude'],axis=1,inplace=True)
pred_columns.drop(['ID'],axis=1,inplace=True)
pred_columns.drop(['Name'],axis=1,inplace=True)
pred_columns.drop(['Date'],axis=1,inplace=True)
pred_columns.drop(['Time'],axis=1,inplace=True)

prediction_var = pred_columns.columns

print(list(prediction_var))


#...............................................   PREDICTION AND TESTING OF DATA VALUES   .........................................................................


train, test = train_test_split(data, test_size = 0.3,random_state=46)

print("\n \n The shape of the Train Data is    :      ",train.shape)
print("\n \n The shape of the Test Data is    :      ",test.shape)


train_X = train[prediction_var]
train_y=train['Status']

test_X= test[prediction_var] 
test_y =test['Status']   

##############################################################################################################


model = tree.DecisionTreeClassifier()
model.fit(train_X,train_y)

pred_y=model.predict(test_X)

df=pd.DataFrame(pred_y,test_y)

print(df)

print(metrics.accuracy_score(pred_y,test_y))
