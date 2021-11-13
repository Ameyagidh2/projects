import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
os.chdir(r"A:\Data science")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#scalar = StandardScaler()
#df=pd.read_csv("Loan_logistic.csv")
#print(df)
#print(df.isnull().sum())
#df.dropna(inplace=True)
#df = pd.get_dummies(df, drop_first=True)
#print(df.isnull().sum())
#df['loanamt'] = scalar.fit_transform(df[['loanamt']])
#df['income'] = scalar.fit_transform(df[['income']])
#print(df)
#x=df.drop(columns="status_Y").values
#y=df["status_Y"].values
#lr=LogisticRegression()
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)
#lr.fit(x_train,y_train)
#pred_y=lr.predict(x_test)
#print(pred_y)
#print(y_test)
#cm=confusion_matrix(y_test,pred_y)
#print(cm)
#score=lr.score(x_test,y_test)
#print(score)

#KNN
#from sklearn.preprocessing import LabelEncoder
#from sklearn.neighbors import KNeighborsClassifier
#df=pd.read_csv("tissue.csv")
#print(df)
#x=df.iloc[:,0:2]   #x=df[["x1","x2"]].values
#print(x)
#y=df.iloc[:,2].values
#print(y)
#le=LabelEncoder()
#y=le.fit_transform(y)
#classifier=KNeighborsClassifier(n_neighbors=3,metric="euclidean")
#print(classifier.fit(x,y))
#print(classifier.predict([[3,7]]))

#SVM
#df=pd.read_csv("Social_Network_Ads.csv")
#print(df)

# Unsupervised Learning Kmeans
#df=pd.read_csv("k means 1.csv")
#print(df)
#plt.scatter(df["X"],df["Y"])
#plt.show()
from sklearn.cluster import KMeans
#km=KMeans(n_clusters=2)
#y_pred=km.fit_predict(df)
#print(y_pred)
#df["Cluster"]=y_pred
#print(df)
#df1=df[df.Cluster==0]
#df2=df[df.Cluster==0]
#plt.scatter(df1["X"],df1["Y"],color="blue")
#plt.scatter(df2["X"],df2["Y"],color="red")
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="purple",marker="*",label="Centroid")
#plt.xlabel("X1")
#plt.ylabel("X2")
#plt.legend()
#plt.show()
#c1=km.cluster_centers_[:,0]#for x1
#print(c1)
#c2=km.cluster_centers_[:,1]#for x2
#print(c2)

#Decision Tree
#df=pd.read_csv("AdultIncome.csv")
#print(df)
#a=df.isnull().sum()
#print(a)
#df.drop(columns="race",inplace=True)
#df=pd.get_dummies(df,drop_first=True)
#print(df)
#x=df.iloc[:,:-1]
#y=df.iloc[:,-1]
#print(x)
#print(y)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)

from sklearn.tree import DecisionTreeClassifier
#dtc=DecisionTreeClassifier(random_state=0)
#dtc.fit(x_train,y_train)
#y_pred=dtc.predict(x_test)
#print(y_pred)
from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#score=dtc.score(x_test,y_test)
#print(cm)
#print(score)

#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
#rfc=RandomForestClassifier(random_state=0)
#rfc.fit(x_train,y_train)
#y_pred=rfc.predict(x_test)
#cm2=confusion_matrix(y_test,y_pred)
#score2=rfc.score(x_test,y_test)

#NLP
import re
import nltk
nltk.download('all')
#from nltk.stem.porter import  import porterstemmer
from nltk.corpus import stopwords
df=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)
print(df)
corpus=[]
for i in range(0,1000):
    review=re.sub("[^a-zA-Z] "," df['Review'][i])
    review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.steam(word) for word in review if not word in set(stopwords.words("english"))]
    review = ''.join(review)
    corpus.append(review)
    print(corpus)

#from sklearn.feature import