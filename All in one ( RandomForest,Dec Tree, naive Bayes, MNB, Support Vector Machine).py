import pandas as pd 
import matplotlib.pyplot
import random
import seaborn as sns 

from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm    		


data = pd.read_csv('C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 9/Data Set/pacific.csv')

print(data.head(6))

data.Status = pd.Categorical(data.Status)
data['Status'] = data.Status.cat.codes

print(data.head())

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


train, test = train_test_split(data, test_size = 0.3)

print("\n \n The shape of the Train Data is       :                 ",train.shape)
print("\n \n The shape of the Test Data is         :                ",test.shape)


train_X = train[prediction_var]
train_y= train['Status']

test_X= test[prediction_var] 
test_y =test['Status'] 


#RandomForest classifier

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction=model.predict(test_X)

print("\n \n Accuracy score of the RANDOM FORSET ALGORITHM is                 :                    ",metrics.accuracy_score(prediction,test_y))


#Decision Tree

model = tree.DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)


print("\n \n Accuracy score of the DECISION TREE ALGORITHM is                 :                    ",metrics.accuracy_score(prediction,test_y)) 


# Support Vector Machine

model = svm.SVC(kernel='linear')
model.fit(train_X,train_y)

predicted= model.predict(test_X)
print(" \n \n Accuracy score of the SVM ALGORITHM  is                 :                    ",accuracy_score(test_y, predicted),"\n \n")


# Navie Bayes


gnb = GaussianNB(priors=None, var_smoothing=1e-09)

y_pred_gnb = gnb.fit(train_X, train_y,sample_weight=None)
target_pred = y_pred_gnb.predict(test_X)


cnf_matrix_gnb = confusion_matrix(test_y,target_pred)

print("\n \n The Confusion Matrix is     :         \n \n ",cnf_matrix_gnb)

print("\n \n The accuracy score of the given data using Navie Bayes Theorm is             :                  ",metrics.accuracy_score(target_pred,test_y),"\n \n")


labels = gnb.predict(test_X)
mat = confusion_matrix(test_y, target_pred)
print(test_y.shape)
print(target_pred.shape)
print(labels.shape)
print(test_X.shape)
print(train_X.shape)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels=prediction_var, yticklabels=prediction_var)

matplotlib.pyplot.show()



