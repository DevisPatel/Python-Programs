Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> str=input("Enter the number you want to get factorial of that number  :  ")
Enter the number you want to get factorial of that number  :  6
>>> y=int(str)
>>> x=1
>>> for i in range(0,y):
	i=i+1
	x=x*i
	if(i==y):print("Factorial of the number ",y,' is :','\t','\t',x)

	
Factorial of the number  6  is : 	 	 720
>>> 
