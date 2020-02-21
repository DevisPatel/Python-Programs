import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.impute import SimpleImputer



data = pd.read_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 7\Data Set\horse.csv")

print("\n \n The data in HORSE.CSV  file are successfully loaded. \n \n")

file=pd.DataFrame(data)

file1=file.iloc[:]

imp = SimpleImputer(strategy="most_frequent")

imp2=imp.fit_transform(file1)

print(imp2)      


imp3=pd.DataFrame(imp2)

imp3.to_csv(r"C:\Users\Devis Patel\AppData\Local\Programs\Python\Python37\Programs\Assignment Programs\Day 7\Data Set\horse.csv",)

print("\n \n Data successfully updated. \n \n ")
