
list=[1,2,3,4,'apple','banana','mango','cherry']

print(list)





#    Conctenation
print(list+['guava','chickoo'])

#    Repetition

print(list*3)

#    Indexing

print(list[0])
print(list[1])
print(list[3])


#    Membership Checking

print('potatto' in list)
print('mango' in list)

#    Slicing Operation

print(list[1:4])


#    Delete Item

del(list[0],list[1])

print(list)




#     Pop and Remove Operation


print(list.pop(2))
print(list.pop(0))


list.remove(4)
list.remove('cherry')

print(list)




