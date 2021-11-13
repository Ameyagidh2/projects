import os
os.chdir(r"A:\Data science")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
#df=pd.read_csv("Salary_Data.csv")
#print(df)
#real_x=df['YearsExperience'].values
#real_y=df.iloc[:,1].values
#real_x=real_x.reshape(-1,1)
#real_y=real_y.reshape(-1,1)

#training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
#print(testing_x)
#lin=LinearRegression()
#lin.fit(training_x,training_y)
#pred_y=lin.predict(testing_x)
#print(pred_y)
#print(testing_y)
#error=pred_y-testing_y
#print(error)
#a=lin.predict([[2]])
#print(a)
#print(lin.coef_)#slope
#print(lin.intercept)

#df=pd.read_csv('FuelConsumptionCo2.csv')
#print(df)
#real_x=df["FUELCONSUMPTION_COMB_MPG"].values
#real_x=df["CO2EMISSIONS"].values
#lin=LinearRegression()
#lin.fit(training_x,training_y)
#pred_y=lin.predict(testing_x)


#df=pd.read_csv("HW.csv")
#print(df)
#x=df.iloc[:,0].values.reshape(-1,1)
#y=df.iloc[:,1].values.reshape(-1,1)
#lin=LinearRegression()
#lin.fit(x,y)
#pred_y=lin.predict(x)
#a=np.concatenate((y,pred_y),axis=1)
#print(a)
#error=y-pred_y
#print(error)
#print(np.abs(error))
#a=metrics.mean_squared_error(y,pred_y)
#print(a)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#df=pd.read_csv("Advertising.csv")
#df.columns
#print(df)
#x=df["TV"].values.reshape(-1,1)
#y=df["Sales"].values.reshape(-1,1)
#training_x,training_y,testing_x,training_y=train_test_split(x,y,test_size=0.2,random_state=0)
#lin.fit(x,y)
#pred_y=lin.predict(x)
#lin.fit(training_x,training_y)
#pred_y=lin.predict(testing_x)

#lin=LinearRegression()
#lin.fit(x,y)
#pred_y=lin.predict(x)
#print(pred_y)
#a=metrics.mean_squared_error(y,pred_y)
#print(a)
#print(df.corr())
#sns.heatmap(df.corr(),annot=True)
#plt.show()

# Multi feature linear regression
import numpy as np
df=pd.read_csv("Advertising.csv")
x=df[["TV","Radio"]].values
y=df["Sales"].values.reshape(-1,1)
lin=LinearRegression()
lin.fit(x,y)
pred_y=lin.predict(x)
print(pred_y)
a=metrics.mean_squared_error(y,pred_y)
print(a)
sns.heatmap(df.corr(),annot=True)
plt.show()

#Classification

#df=pd.read_csv("Loan_logistic.csv")
#print(df)
#a=df.isnull().sum()
#print(a)
#df=df.dropna()
#print(df)
#a=df.isnull().sum()
#print(a)
#df=df.drop(columns="gender")
#print(df)
#Make 1 and 0 to true and false
#df=pd.get_dummies(df,drop_first=True)
#print(df)
# Standardization
from sklearn.preprocessing import StandardScaler
#scalar = StandardScaler
#df["income"]=scalar.fit(df[["income"] ])
#df["loanamt"]=scalar.fit_transform(df[["loanamt"]])
#y=df["status_Y"].values
#x=df.drop(columns="status_Y").values
from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)
from sklearn.linear_model import LogisticRegression
#lr=LogisticRegression()
#lr.fit(x_train,y_train)
#pred_y=lr.predict(x_test)
from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,pred_y)
#print(cm)
#score=lr.score(x_test,y_test)
#print(score)