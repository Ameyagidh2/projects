import os
os.chdir(r"A:\Data science")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Titanic (1).csv")
print(df)
a=df["Survived"].value_counts()
#print(a)
s=pd.pivot_table(df, index="Survived", aggfunc="count")
#print(s)
#sns.catplot(x="Sex",hue="Pclass",kind="count",data=df)
#sns.violinplot(x="Sex",y="Age",hue="Survived",data=df,split=True)
plt.show()
z1=[1,np.nan,np.nan,4,5,np.nan,7,np.nan]
z2=[1,1,1,1,2,2,2,2]
df1=pd.DataFrame({'Cricket':z1,
                  "Tennis":z2})
#print(df1)
df1.to_csv("Missing eg.csv",index=False)
#print(df1.fillna(value=0))
#print(df1.fillna(value=df1["Cricket"].mean()))
#Group by function   ,inplace=True
print(df1["Cricket"].fillna(df1.groupby(["Tennis"])['Cricket'].transform(np.mean)))



