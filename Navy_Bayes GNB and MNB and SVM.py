import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import classification_report
from sklearn import svm



data = pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 9\Data Set\run_or_walk.csv")

print("\n \n Data has been successfully loaded and the columns in the data set are     :    \n \n \t \t \t ",data.columns,'\n \n')

file=pd.DataFrame(data)

x=['acceleration_x','acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']


##################################################################################################################

gnb = GaussianNB(priors=None, var_smoothing=1e-09)


train, test = train_test_split(data, test_size = 0.3,random_state=8)

print("\n \n The shape of the Train data is           :              ",train.shape)
print("\n \n The shape of the Train data is           :              ",test.shape)


train_X = train[x]
train_y= train['activity']


test_X= test[x] 
test_y =test['activity']   

y_pred_gnb = gnb.fit(train_X, train_y,sample_weight=None)
target_pred = y_pred_gnb.predict(test_X)


cnf_matrix_gnb = confusion_matrix(test_y,target_pred)

print("\n \n The confusion matrix is     :         \n \n ",cnf_matrix_gnb)

print("\n \n The accuracy score of the given data using Navie Bayes Theorm is             :                  ",metrics.accuracy_score(target_pred,test_y),"\n \n")


############################      CLASSIFICATION REPORT     ##########################################################

visualizer = ClassificationReport(gnb, x=x, support=True)

visualizer.fit(train_X, train_y) 
visualizer.score(test_X,test_y)
g = visualizer.poof()           


###########################################  SUPPORT VECTOR MACHINE   ###########################################


model=svm.SVC(kernel='linear')

model.fit(train_X,train_y)

pred_svc=model.predict(test_X)

print("\n \n Accuray of the given DATA SET using the SUPPORT VECTOR MACHINE is      :          ",accuracy_score(test_y,pred_svc),'\n \n')

