import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file1=pd.read_csv("C:/Users/Devis Patel/AppData/Local/Programs/Python/Python37/Programs/Assignment Programs/Day 6/Data Set/FyntraCustomerData.csv")

file11=pd.DataFrame(file1)

corr= file1.corr()

sns. heatmap(corr,square=True,cmap="BuGn")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


g = sns.jointplot(x="Time_on_App", y="Avg_Session_Length", data=file1)
g.set_axis_labels("X-Axis", "Y-Axis")
plt.show()


'''


seaborn.jointplot(x, y, data=None, kind='scatter', stat_func=None, color=None, height=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)
¶

kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

'''
