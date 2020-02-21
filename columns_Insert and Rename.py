import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file1=pd.read_csv("C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 6/Data Set/prisoners.csv")

file11=pd.DataFrame(file1)

file11.rename(columns={"No. of Inmates benefitted by Elementary Education":"Inmate1","No. of Inmates benefitted by Adult Education":"Inmate2","No. of Inmates benefitted by Higher Education":"Inmate3","No. of Inmates benefitted by Computer Course":"Inmate4"},inplace=True)

file11.insert(6,'total_benefitted',1000)

