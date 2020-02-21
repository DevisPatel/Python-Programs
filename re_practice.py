import re
import os


xx ="guru99,education is fun"

'''
print("\n",re.sub(r'\d','%',"12+12=24"))

print("\n",re.sub(r'\D','*',"3+12=24"))

print( "\n",re.findall(r"^\w+", xx))

print("\n",(re.split(r'\s','we are splitting the words')))

print("\n",(re.split(r's','split the words')))

'''

print("\n",(re.sub(r'\w','*','This is a test 0 or is it?')))

print("\n",(re.sub(r'\W','*','This is a test 0 or is it?')))

print("\n",(re.sub(r'\s','*','This is a test 0 or is it?')))

print("\n",(re.sub(r'\S','*','This is a test 0 or is it?')))

print("\n",(re.sub(r'the(?=cat)','*','the dog and the cat')))

print("\n",(re.sub(r'(?<=)the','*','the dog and the cat')))

print("\n",(re.sub(r'(?i)ab','*','ab AB  cdabacbcabd abcdefg')))




'''

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


list = ["guru99 get", "guru99 give", "guru Selenium"]

for element in list:
    z = re.match("(g\w+)\W(g\w+)", element)

if z:
    print((z.groups()))
    


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
patterns = ['software testing', 'guru99']
text = 'software testing is fun?'


for pattern in patterns:

    print('\n \n \t \t Looking for "%s" in "%s" ->' % (pattern, text), end=' ')

    if re.search(pattern, text):
        print('found a match!')
    else:
        print('no match')

abc = 'guru99@google.com, careerguru99@hotmail.com, users@yahoomail.com'

emails = re.findall(r'[\w\.-]+@[\w\.-]+', abc)

for email in emails:
    print('\n \t \t ',email)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


xx1 = """guru99 careerguru99	selenium"""


k1 = re.findall(r"^\w", xx)

k2 = re.findall(r"^\w", xx, re.MULTILINE)

print('\n \t \t ',k1)

print('\n \t \t ',k2)


'''
