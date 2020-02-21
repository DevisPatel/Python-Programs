import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt


#table=pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 4\Data Set\save.csv")
#print(table)


plt.bar([1,40,7],[4,12,16])

#plt.plot([r**2 for i in r])

#plt.plot([1264,0.655676456634422,-300,47,5.563,-60])

#plt.plot([1212,-8868,87.6756])

#plt.plot([112,-888,37.6756,-65.09897])

#plt.plot([126,0.652,-300,47,5.53,-50])

plt.show()







'''

table=[{'a':10,'b':20},{'a':30,'c':40},{'a':50,'b':60},{'a':70,'d':80}]

t1=pd.DataFrame(table)

print(t1)

num=rd.randrange(0,50)
print(num)

r=rd.randrange(0,100,25)
print(r)
q=rd.randint(0,50)
print(q)



#.......................................................................


list1=[1,2,3,4,5,6]

a=np.array(list1)

b=np.array([[2,3,4],[1,3,6],[4,2,7]])

for i in range(0,4):

    for j in range(0,4):

        c=np.array([[i],[j]])


d=np.zeros((4,4))

print(a,'\n \n',b,'\n \n',c,'\n \n',d)

print(d[3][2])

#.............................................................................



d=np.zeros((8))

d1=d.reshape((2,2,2))

print(d1)

e=np.arange(2,80)

print(e,'\n \n',e[6],'  ',e[14])


x=slice(1,15,3)

print("\n \n \t Slicing in Array is  \n  \t ",e[x])





#..................................................................................


a=np.array([1,2,3,4])
print(a)

a2=np.array([[1,2,3,4],[5,6,7,8]])

print(a2)


list1=[1,2,3,4,5,6,7,8,9]

list2=[[1,2,3],[4,5,6]]

a3=np.array(list1)
print(a3,'    ',a3[5])

'''



