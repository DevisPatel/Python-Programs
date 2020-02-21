from sklearn.preprocessing import  PolynomialFeatures
from sklearn import linear_model as lm

x=[[0.48,0.59],[0.99,0.35]]

vector=[109.85,158.69]

print("\n \n Value of original vector using POLYNOMIAL REGRESSION is     :       ",vector)

predict=[[0.46,0.54],[0.96,0.14]]

ploy=PolynomialFeatures(degree=2)

x1=ploy.fit_transform(x)
predict1=ploy.fit_transform(predict)

clf=lm.LinearRegression()

clf.fit(x1,vector)

clf2=clf.predict(predict1)

print("\n \n Value of predicted vector using POLYNOMIAL REGRESSION is     :       ",clf2,'\n \n')
