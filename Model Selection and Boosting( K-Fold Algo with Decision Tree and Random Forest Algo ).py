import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import*
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 14\Data Set\glass.csv")


file=pd.DataFrame(data)

types=[]

for i in range(0,len(data)):

    types.append(file['Type'][i])


label=['Type 1','Type 2','Type 2','Type 4','Type 5','Type 6','Type 7']

plt.plot(types,'r')

plt.bar(types,len(label))
plt.grid(True)
plt.ylabel('Types of Glasses', fontsize=10)
plt.xlabel('Number of Glasses', fontsize=10)
plt.title("Bar plot of different types of glasses")

plt.show()

##########################################################################################################


random.seed(2)

pred_columns=data.iloc[:,0:9]


prediction_var=pred_columns.columns

print('\n \n ',list(prediction_var))


train, test = train_test_split(data, test_size = 0.3,random_state=4)

print("\n \n The shape of the Train Data is    :      ",train.shape)
print("\n \n The shape of the Test Data is    :      ",test.shape)


train_X = train[prediction_var]
train_y=train['Type']

test_X= test[prediction_var] 
test_y =test['Type']   


model = tree.DecisionTreeClassifier()
model.fit(train_X,train_y)

pred_y=model.predict(test_X)

print("\n \n The accuracy score is using DECISION TREE ALGORITHM is      :                                        ",metrics.accuracy_score(pred_y,test_y))


##################################      Accuracy using K-FOLD      ##############################################################


x = data.iloc[:,0:9]
y = data['Type']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=4)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

print("\n \n The accuracy score  using K-FOLD ALGORITHM compare with DECISION TREE ALGO is      :                                        ",metrics.accuracy_score(y_test,y_pred))


######################################          RANDOM FOREST using 10 K-FOLD Split          ####################################################

model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print("\n \n The accuracy score of RANDOM FOREST ALGORITHM    is       :          ",metrics.accuracy_score(prediction,test_y))



############################################################################################################################################


x = data.iloc[:,0:9]
y = data['Type']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=4)

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

print("\n \n The accuracy score using K-FOLD ALGORITHM compare with RANDOM FOREST ALGO is      :                                        ",metrics.accuracy_score(y_test,y_pred),"\n \n ")


