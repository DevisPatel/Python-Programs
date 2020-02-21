import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

file1=pd.read_csv("C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 6/Data Set/FyntraCustomerData.csv")

file11=pd.DataFrame(file1)

x=file11.iloc[:,3:7]
y=file11['Yearly_Amount_Spent']


#......................................   CORRELATION ......................................................................................................................

'''
corr= file1.corr()

sns. heatmap(corr,square=True,cmap="YlGnBu")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

'''

#.............................................   TRAIN AND TEST DATA ..................................................................................................

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=8)

#print("\n \n The training data values of X and Y are   :         ",x_train,"  ,  ",y_train)

lm=LinearRegression()

model=lm.fit(x_train,y_train)

#.............................................   PREDICTION OF DATA AND INAL OUTPUT    ...................................................................


pred_y = lm.predict(x_test)

file11=pd.DataFrame(pred_y,y_test)

print(file11)

#print(metrics.accuracy_score(pred_y,y_test))

plt.scatter(y_test,pred_y)
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")

plt.show()






