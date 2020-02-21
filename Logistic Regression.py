import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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




data = pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 6\Data Set\data.csv",header=0)

print(data.head(6))

data.info()

# now we can drop this column Unnamed: 32
data.drop("Unnamed: 32",axis=1,inplace=True) 
data.columns

# like this we also don't want the Id column for our analysis
data.drop("id",axis=1,inplace=True)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


data.describe()

'''
#...................................................   DATA VISUALIZATION  ........................................................................................................

sns.countplot(data['diagnosis'],label="Count")
plt.show()

corr = data.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True,cmap= 'BuGn')
plt.show()
'''
#...............................................   PREDICTION AND TESTING OF DATA VALUES   .........................................................................

#Based on corrplot let's select some features for the model ( decision is made in order to remove collinearity)
prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
# now these are the variables which will use for prediction


train, test = train_test_split(data, test_size = 0.3,random_state=46)# in this our main data is splitted into train and test

print(train.shape)
print(test.shape)


train_X = train[prediction_var]# taking the training data input
train_y=train.diagnosis# This is output of our training data

# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test.diagnosis   #output value of test data


logistic = LogisticRegression()
logistic.fit(train_X,train_y)
pred_y=logistic.predict(test_X)

print(metrics.accuracy_score(pred_y,test_y))



'''
########################  DECISION TREE ALGORITHM  ####################################################################


clf = DecisionTreeClassifier(random_state=0)
cross_val_score(clf, train_X, train_y, cv=10)
clf.fit(train_X,train_y, sample_weight=None, check_input=True, X_idx_sorted=None)
clf.get_params(deep=True)
clf.predict(test_X, check_input=True)
clf.predict_log_proba(test_X)
clf.predict(test_X,check_input=True)

print(clf.score(test_X,test_y, sample_weight=None))

'''







