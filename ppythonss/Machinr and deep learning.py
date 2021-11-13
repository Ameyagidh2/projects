import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib. pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score,roc_auc_score
import statsmodels.api as sn
import statsmodels.discrete.discrete_model as sm
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
os.chdir(r'A:\Data science\datasets for ml deep learning')
df=pd.read_csv("House_Price.csv",header=0)
df1=pd.read_csv("House-Price1.csv",header=0)
x_mult=df1.loc[:,df1.columns !='Sold']
y=df1['Sold']
x_train,x_test,y_train,y_test=train_test_split(x_mult,y,test_size=0.2,random_state=0)
'''
x1=df.drop("price",axis=1)
y1=df['price']
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
#print(df.head())
#print(df.shape)
#print(df)
#Univariate analysis ie studying all the single variable variation with an output
#print(df.columns)
#print(df.describe())#shows mean median std etc
#here n_host_beds has count <506
#crime rate has outliers as mean - median is different and bit for age
# for non categorial use joint plots ie histograms
#sns.jointplot(x="n_hot_rooms",y="price",data=df)
#plt. show()
#2 outliers present
#sns.jointplot(x="rainfall",y="price",data=df)
#plt. show()
#1 outlier
#price tends to be clogged on the area of higher to medium rainfall

# for categorial variables use bar chart as they have labels as string form
#eg airport,bus_ter,water body use count plots eg bar plots
#sns.countplot(x='bus_ter',data=df)
#plt.show()
#You get a rectangular graph of just 1 yes value and nothing else so this is redundant variable and hence can be ignored in our analysis


#finding outliers
#df.info()
a1=np.percentile(df.n_hot_rooms,[99])[0]
#print(a1)
uv=a1*3
#Now replace data frame with vallues greater than uv
df.n_hot_rooms[(df.n_hot_rooms>uv)]=uv#since mean and max have large difference
#print(df.describe())
#changing lower limit of rainfall to 0.1 times percentile as outlier is 3 changed to 6
a2=np.percentile(df.rainfall,[1])[0]
print(a2)
lv=a2*0.3
df.rainfall[(df.rainfall<lv)]=lv
#print(df.describe())#min changed from 3 to 6
#print(df)
#Missing values treatment
#we find that 498 values in n_hos_beds
df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())
#print(df.info())
#Bivariate variable transformation
df['avg_dist']=(df.dist1+df.dist2+df.dist3+df.dist4)/4
#print(df)
df.crime_rate=np.log(1+df.crime_rate)
df2=df
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
del df['bus_ter']
df=pd.get_dummies(df)
#print(df.describe())
del df['airport_NO']
del df['waterbody_None']
#print(df)
#correlation matrix in data know wh
#print(df.corr())
#parks and air quality high correlation hence remove 1 which is not having higher coefficient with our dependent variable price
del df['parks']
#print(df.info())
#rainfall and n_hot_beds not very imporant as low r with x dependent variable

# Linear Regression
#Linear regression rsquare is how close data fitted wrt regression line,std error spread of data Relational standard error RSE is u]the absolute distance difference lsm method of the points from the regression line., t probability gives if low the ha that is anti hypothesis value is small then good ie p{<t}should be close to 0 and estimates are the intercetps b0 and slope B1
'''
'''
#Method 1:-
import statsmodels.api as sm
x=sm.add_constant(df['room_num'])
lm=sm.OLS(df['price'],x).fit()
print(lm.summary())
#Method2:-

from sklearn.linear_model import LinearRegression
import sklearn.metrics
lm=LinearRegression()
x=df[['room_num']]
y=df['price']
lm.fit(x,y)
print(lm.intercept_)
print(lm.coef_)
#plot simple linear regression
#help(sns.jointplot)
#sns.jointplot(x=df['room_num'],y=df['price'],data=df,kind="reg")
#plt.show()
print(lm.score(x,y))#R^2
#print(lm.predict(x))#from x input get y as output
print()
'''
'''
#Multiple regression
from sklearn.linear_model import LinearRegression
#m1
import statsmodels.api as sm
x1=df.drop('price',axis=1)#All columns except price
x1=sm.add_constant(x1)
y1=df['price']
lm=sm.OLS(y1,x1).fit()  #Y first then x
print(lm.summary())

#m2
from sklearn.linear_model import LinearRegression
x1=df.drop('price',axis=1)
y1=df['price']
lm= LinearRegression()
lm.fit(x1,y1)
print(lm.coef_,lm.intercept_)
'''
'''
#Test train split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
lm=LinearRegression()
x1=df.drop("price",axis=1)
y1=df['price']
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
lm.fit(x_train,y_train)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

y_test_pred=lm.predict(x_test)
#print(y_test_pred)

y_train_pred=lm.predict(x_train)
#print(y_train_pred)
#find the prediction rsquare value of train and test to validate that train more data better the r coefficient value
rtest=r2_score(y_test,y_test_pred)
print(rtest)
rtrain=r2_score(y_train,y_train_pred)
print(rtrain)
'''
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
x1=df.drop("price",axis=1)
y1=df['price']
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2,random_state=0)

#Lasso and Ridge regression
#Ridge regression
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
#Make x in standardized form for lasso and ridge
scaler=preprocessing.StandardScaler().fit(x_train)
x_train_standard=scaler.transform(x_train)
x_test_standard=scaler.transform(x_test)
lm_r=Ridge(alpha=0.5)
lm_r.fit(x_train_standard,y_train)
y_test_standard=lm_r.predict(x_test_standard)
r22=r2_score(y_test,y_test_standard)
print(r22)
#Now this r2_score needs to find best optimal value of alpha use params method
params_ranges=np.logspace(-2,8,100)#100 nos from 10^-2 to 10^8
#help(Ridge)
#help(validation_curve)
train_scores,test_scores=validation_curve(Ridge(),X=x_train_standard,y=y_train,param_name='alpha',param_range=params_ranges,scoring='r2')
#finding max of this test and train score and corresponding alpha value

train_mean=np.mean(train_scores,axis=1)
print(max(train_mean))#This is alpha value for best fit of train data similarly for test data
test_mean=np.mean(test_scores,axis=1)
print(max(test_mean))
#sns.jointplot(x=np.log(params_ranges),y=test_mean)
#plt.show()
loc_alpha_in_array=np.where(max(test_mean)==test_mean)
print(loc_alpha_in_array)
lm_best_fit=Ridge(alpha=params_ranges[31])
lm_best_fit.fit(x_train,y_train)
r222=r2_score(y_test,lm_best_fit.predict(x_test))#after fitting best model this is prediction value
print(r222)
from sklearn.linear_model import Lasso
lm_lasso=Lasso(alpha=0.4)
'''

#Classification
#EDD of Data
#print(df.head())
#print(df1.shape)
#print(df1.info())
#print(df1.describe())
#print(df.describe())
#help(sns.jointplot())
#Box plot
#sns.boxplot(y='rainfall',data=df1)
#plt.show()
#Scatter plot
#sns.jointplot(x='n_hot_rooms',y='price',data=df1)
#plt.show()
#Bar plot
#sns.countplot(x='air_qual',data=df1)
#plt.show()
'''
df1=pd.read_csv("House-Price1.csv",header=0)
a1=np.percentile(df.n_hot_rooms,[99])[0]
#print(a1)
uv=a1*3
#Now replace data frame with vallues greater than uv
df.n_hot_rooms[(df.n_hot_rooms>uv)]=uv#since mean and max have large difference
#print(df.describe())
#changing lower limit of rainfall to 0.1 times percentile as outlier is 3 changed to 6
a2=np.percentile(df.rainfall,[1])[0]
print(a2)
lv=a2*0.3
df.rainfall[(df.rainfall<lv)]=lv
#print(df.describe())#min changed from 3 to 6
del df1['bus_ter']
df1['avg_dist']=(df1['dist1']+df1['dist2']+df1['dist3']+df1['dist4'])/4
del df1['dist1']
del df1['dist2']
del df1['dist3']
del df1['dist4']
df1=pd.get_dummies(df1)
del df1['waterbody_None']
del df1['airport_NO']
#print(df1.describe())
df1['n_hos_beds']=df1.fillna(df1["n_hos_beds"].mean())
print(df1)

#Logistic Regression
#Single varibale
'''
'''
from sklearn.linear_model import LogisticRegression
clf_lr=LogisticRegression()
x_single=df1[['price']]
y=df1['Sold']
clf_lr.fit(x_single,y)
print(clf_lr.coef_)
print(clf_lr.intercept_)


import statsmodels.api as sn
import statsmodels.discrete.discrete_model as sm
#For stats model use output first then input
x_single_standard=sn.add_constant(x_single)
stats_var=sm.Logit(y,x_single_standard).fit()
#print(stats_var.summary())
'''
'''
#Logistic Regression for multi variable logistic regression
from sklearn.linear_model import LogisticRegression
clf_lr2=LogisticRegression(max_iter=100)
#x_mult=df1.drop('Sold',axis=1)
x_mult=df1.loc[:,df1.columns!='Sold']

y=df1['Sold']
clf_lr2.fit(x_mult,y)
print(clf_lr2.intercept_)
print(clf_lr2.coef_)

#Confusion Matrix
x_pre=clf_lr2.predict(x_mult)
proba=clf_lr2.predict_proba(x_mult)
print(proba)

y_pred=clf_lr2.predict(x_mult)
#print(y_pred)
y_pred_03=(clf_lr2.predict_proba(x_mult)[:,1]>=0.3)#Chnaging threshold for 0 and 1 from 0.5 to 0.3
#print(y_pred_03)
print(confusion_matrix(y,y_pred))
print(confusion_matrix(y,y_pred_03))

print(recall_score(y,y_pred))
print(precision_score(y,y_pred))
print(roc_auc_score(y,y_pred))
'''
'''
print(recall_score(y,y_pred_03))
print(precision_score(y,y_pred_03))
print(roc_auc_score(y,y_pred_03))
'''


'''
#Not working
import statsmodels.api as sn
import statsmodels.discrete.discrete_model as sm
x_mult=df1.loc[:,df1.columns!='Sold']
#For stats model use output first then input
x_mult_standard=sn.add_constant(x_mult)
y=df1['Sold']
stats_var1=sm.Logit(y,x_mult_standard).fit()
print(stats_var1.summary())
'''
'''
# By default the 0.5 value is the threshold changing that we use pred_prob function
#Linear discriminant analysis (LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
x_mult=df1.loc[:,df1.columns!='Sold']
print(x_mult)
y=df1['Sold']
clf_lda=LinearDiscriminantAnalysis()
clf_lda.fit(x_mult,y)
y_predicted=clf_lda.predict(x_mult)
#print(y_predicted)
#print(confusion_matrix(y,y_predicted))

#Perfect classification model Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
clf_lr=LogisticRegression()
x_mult=df1.loc[:,df1.columns !='Sold']
y=df1['Sold']
x_train,x_test,y_train,y_test=train_test_split(x_mult,y,test_size=0.2,random_state=0)
clf_lr.fit(x_train,y_train)
print(x_train.shape)
print(x_test.shape)
y_predict_Logistic_Regression=clf_lr.predict(x_test)
print(y_predict_Logistic_Regression)
print("Confusion matrix:LDA")
print(confusion_matrix(y_test,y_predict_Logistic_Regression))
print(accuracy_score(y_test,y_predict_Logistic_Regression))

#Perfect LDA Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf_lda=LinearDiscriminantAnalysis()
clf_lda.fit(x_train,y_train)
y_predict_LinearDiscriminantAnalysis=clf_lda.predict(x_test)
print("Confusion matrix:LDA")
print(confusion_matrix(y_test,y_predict_LinearDiscriminantAnalysis))
print(accuracy_score(y_test,y_predict_LinearDiscriminantAnalysis))

#perfect KNN k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
scaler=preprocessing.StandardScaler().fit(x_train)
#Standardized inputs of x to the k neighborsclassifier
x_train_scale=scaler.transform(x_train)

scaler=preprocessing.StandardScaler().fit(x_test)
x_test_scale=scaler.transform(x_test)

clf_knn=KNeighborsClassifier(n_neighbors=7)
clf_knn.fit(x_train_scale,y_train)
y_pred_knn=clf_knn.predict(x_test_scale)
print("Confusion matrix knn")
print(confusion_matrix(y_test,y_pred_knn))
print(accuracy_score(y_test,y_pred_knn))

#Loop for geting optimized k value
params={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
from sklearn.model_selection import GridSearchCV
grid_loopk=GridSearchCV(KNeighborsClassifier(),params)
grid_loopk.fit(x_train_scale,y_train)
print(grid_loopk.best_params_)
optimized_knn=grid_loopk.best_estimator_
y_optimized_knn=optimized_knn.predict(x_test_scale)
print(confusion_matrix(y_test,y_optimized_knn))
#Comparing results from three classifiers


#Perfect linear regression Summary using fit of linear regression
import statsmodels.api as sn
import statsmodels.discrete.discrete_model as sm
#For stats model use output first then input
x_train=sn.add_constant(x_train)
model = sn.OLS(y_train,x_train)
results = model.fit()
print(results.summary())
#x_train_scale=sn.add_constant(x_train_scale)
#stats_var=sm.Logit(y_train,x_train_scale).fit()
#print(stats_var.summary())
'''

'''
#Decision tree Regression
from sklearn import tree
#from Ipython.display import Image
#import pydotplus
df_tree=pd.read_csv('Movie_regression.csv',header=0)
#print(df_tree.head)
#print(df_tree.shape)
#print(df_tree.describe()
print(df_tree.info())
#Time_taken has null values
df_tree=df_tree.fillna(value=df_tree['Time_taken'].mean())
df_tree=pd.get_dummies(df_tree,columns=['3D_available','Genre'],drop_first=True)
print(df_tree.info())

from sklearn.model_selection import train_test_split
x=df_tree.loc[:,df_tree.columns!='Collection']
y=df_tree['Collection']
x_tree_train,x_tree_test,y_tree_train,y_tree_test=train_test_split(x,y,random_state=0,test_size=0.2)
print(x_tree_train.shape)

reg_tree=tree.DecisionTreeRegressor(min_samples_split=40)#min_samples_split=40 min_samples_leaf=40   max_depth=3
reg_tree.fit(x_tree_train,y_tree_train)

y_tree_pred=reg_tree.predict(x_tree_test)
print(r2_score(y_tree_test,y_tree_pred))
'''
'''
#print(mean_squared_error(y_tree_test,y_tree_pred))
#Plotting the decision tree
from IPython.display import*
from pydotplus import *

#does not workdot-image-graph 
dot_data=tree.export_graphviz(reg_tree,out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())
from sklearn.tree import plot_tree
plt.figure(figsize=(5,5))
plot_tree(reg_tree, filled=True)
plt.show()
'''
'''
#Decision tree classification
from sklearn import tree
df_clf_tree=pd.read_csv("Movie_classification.csv",header=0)
#missing values
df_clf_tree.Time_taken=df_clf_tree.Time_taken.fillna(value=df_clf_tree.Time_taken.mean())
#print(df_clf_tree.describe())
#dummy creation
df_clf_tree=pd.get_dummies(df_clf_tree,columns=['3D_available','Genre'],drop_first="True")
#print(df_clf_tree.describe())
#test train split
x_clf_tree=df_clf_tree.loc[:,df_clf_tree.columns!='Start_Tech_Oscar']
y_clf_tree=df_clf_tree["Start_Tech_Oscar"]
x_train_clf_tree,x_test_clf_tree,y_train_clf_tree,y_test_clf_tree=train_test_split(x_clf_tree,y_clf_tree,random_state=0,test_size=0.2)

clf_tree=tree.DecisionTreeClassifier(max_depth=3)#min_samples_split=6
clf_tree.fit(x_train_clf_tree,y_train_clf_tree)
y_predicted_clf_tree=clf_tree.predict(x_test_clf_tree)
#y_predicted_clf_tree=clf_tree.predict(x_test_clf_tree)
print("Confusion_matrix of Decision tree classification")
print(confusion_matrix(y_predicted_clf_tree,y_test_clf_tree))

'''
'''
from sklearn.tree import plot_tree
plt.figure(figsize=(5,5))
plot_tree(clf_tree,filled=True)
plt.show()
'''
'''

#Bagging selects all variables to predict the output
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
#df_clf_tree - data set
#X-- x_clf_tree
#Y--y_clf_tree

clf_tree_bagging=tree.DecisionTreeClassifier()
clf_bag=BaggingClassifier(base_estimator=clf_tree_bagging,n_estimators=1000,n_jobs=-1,bootstrap=True,random_state=42)
clf_tree_bagging.fit(x_train_clf_tree,y_train_clf_tree)
y_pred_bag=clf_tree_bagging.predict(x_test_clf_tree)
print("Confusion_matrix of Bagging")
print(confusion_matrix(y_test_clf_tree,y_pred_bag))
#print(accuracy_score(y_test_clf_tree,y_pred_bag))



#random forest choose few certain variables
from sklearn.ensemble import RandomForestClassifier
clf_randomforest=RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=42)#
clf_randomforest.fit(x_train_clf_tree,y_train_clf_tree)
y_predicted_randomforest=clf_randomforest.predict(x_test_clf_tree)
print("Confusion_matrix of RandomForestClassifier")
print(confusion_matrix(y_predicted_randomforest,y_pred_bag))
print(accuracy_score(y_predicted_randomforest,y_pred_bag))


'''
'''
#Grid search for best parameters detection
from sklearn.model_selection import GridSearchCV
params_grid={'max_features':[5,6,7,8],
             'min_samples_split':[2,3]}

grid_search=GridSearchCV(clf_randomforest,params_grid,n_jobs=-1,cv=5,scoring='accuracy')
grid_search.fit(x_train_clf_tree,y_train_clf_tree)
optimum=grid_search.best_estimator_
print(optimum)
print(accuracy_score(y_predicted_randomforest,optimum.predict(x_test_clf_tree)))
print("Best fit for random forest")
print(confusion_matrix(y_predicted_randomforest,optimum.predict(x_test_clf_tree)))
'''
'''
'''
#Boosting
#Gradient boosting
df_boost=pd.read_csv("Movie_classification.csv",header=0)
df_boost.Time_taken=df_boost.Time_taken.fillna(value=df_boost.Time_taken.mean())
#print(df_boost.info())

df_boost=pd.get_dummies(df_boost,columns=['3D_available','Genre'],drop_first="True")
x_boosting_gradient=df_boost.loc[:,df_boost.columns!="Start_Tech_Oscar"]
y_boost_gradient=df_boost["Start_Tech_Oscar"]
x_train_gb,x_test_gb,y_train_gb,y_test_gb=train_test_split(x_boosting_gradient,y_boost_gradient,test_size=0.2,random_state=0)
'''
from sklearn.ensemble import GradientBoostingClassifier
clf_grad_boost=GradientBoostingClassifier(max_depth=1,n_estimators=1000,learning_rate=0.02)
clf_grad_boost.fit(x_train_gb,y_train_gb)
y_pred_gb=clf_grad_boost.predict(x_test_gb)
print('Confusion_matrix of Gradient Boosting')
print(confusion_matrix(y_test_gb,y_pred_gb))
'''
'''
from sklearn.model_selection import GridSearchCV
params_grid={'learning_rate':[0.01,0.03,0.04,0.05,0.06,0.07,0.08,0.09,1]}
grid_search=GridSearchCV(clf_grad_boost,params_grid,cv=5,scoring='accuracy',n_jobs=-1)
grid_search.fit(x_train_gb,y_train_gb)
optimum=grid_search.best_estimator_
print(optimum)
print(confusion_matrix(y_test_gb,optimum.predict(x_test_gb)))
'''
'''
#Ada boost we can use a base estimator using object of random forest
from sklearn.ensemble import AdaBoostClassifier
clf_ada=AdaBoostClassifier(clf_randomforest,learning_rate=0.02,n_estimators=1000)
clf_ada.fit(x_train_gb,y_train_gb)
y_pred_ada=clf_ada.predict(x_test_gb)
print("Confusion matrix for ada classifier")
print(confusion_matrix(y_test_gb,y_pred_ada))
'''
'''
#XG Boost
import xgboost as xgb
#x_train_gb,y_train_gb
clf_xgb=xgb.XGBClassifier(max_depth=5,n_estimators=100,learning_rate=0.3,n_jobs=-1)
clf_xgb.fit(x_train_gb,y_train_gb)
y_pred_xgb=clf_xgb.predict(x_test_gb)
print(confusion_matrix(y_test_gb,y_pred_xgb))
print(accuracy_score(y_test_gb,y_pred_xgb))
#Grid searchcv
from sklearn.model_selection import GridSearchCV
'''
'''
param_test={"max_depth":range(3,6,2),
            "gaama":[0.1,0.2,0.3],
            'sub_sample':[0.8,0.9],
            'colsample_bytree':[0.8,0.9],
            'reg_apha':[1e-2,0.1,1]}
            
param_test={"max_depth":range(3,6,2),
            "gaama":[0.1,0.2,0.3]}
grid_search=GridSearchCV(clf_xgb,param_test,n_jobs=-1,cv=5,scoring='accuracy')
grid_search.fit(x_train_gb,y_train_gb)
optimum=grid_search.best_estimator_
print('best parametors using xgb boosting')
print(confusion_matrix(y_test_gb,optimum.predict(x_test_gb)))
print(accuracy_score(y_test_gb,optimum.predict(x_test_gb)))
'''
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import accuracy_score,confusion_matrix,r2_score,mean_squared_error
from sklearn import svm
from sklearn.model_selection import train_test_split
'''
#Support vector machines
#Regression
#Preprocessing
df_reg=pd.read_csv('Movie_regression.csv',header=0)
df_reg=df_reg.fillna(value=df_reg['Time_taken'].mean())
df_reg=pd.get_dummies(df_reg,columns=['3D_available','Genre'],drop_first=True)
print(df_reg.info())

x=df_reg.loc[:,df_reg.columns!='Collection']
y=df_reg['Collection']
#split
x_reg_train,x_reg_test,y_reg_train,y_reg_test=train_test_split(x,y,test_size=0.2,random_state=0)
#convert x into standard form
sc=StandardScaler().fit(x_reg_train)
x_reg_train_std=sc.transform(x_reg_train)
sc2=StandardScaler().fit(x_reg_test)
x_reg_test_std=sc2.transform(x_reg_test)
#svm regression
reg_svm=svm.SVR(kernel='linear',C=1500)
reg_svm.fit(x_reg_train_std,y_reg_train)
y_svm_reg=reg_svm.predict(x_reg_test_std)
print(mean_squared_error(y_reg_test,y_svm_reg))
print(r2_score(y_reg_test,y_svm_reg))
'''
'''
#Classification
df_class=pd.read_csv('Movie_classification.csv',header=0)
df_class=df_class.fillna(value=df_class['Time_taken'].mean())
df_class=pd.get_dummies(df_class,columns=['3D_available','Genre'],drop_first=True)
print(df_class.info())

x=df_class.loc[:,df_class.columns!='Start_Tech_Oscar']
y=df_class['Start_Tech_Oscar']
#split
x_class_train,x_class_test,y_class_train,y_class_test=train_test_split(x,y,test_size=0.2,random_state=0)
#convert x into standard form
sc=StandardScaler().fit(x_class_train)
x_class_train_std=sc.transform(x_class_train)
sc2=StandardScaler().fit(x_class_test)
x_class_test_std=sc2.transform(x_class_test)
#svm regression
class_svm=svm.SVC(kernel='linear',C=0.01)
class_svm.fit(x_class_train_std,y_class_train)
y_svm_class=class_svm.predict(x_class_test_std)
print(confusion_matrix(y_class_test,y_svm_class))
print(accuracy_score(y_class_test,y_svm_class))

#best fit
from sklearn.model_selection import GridSearchCV
class_svm2=svm.SVC(kernel='linear')
params={'C':[0.001,0.005,0.03,0.05,0.07,0.08,1,1.5,50,100,500,1000,1500,3000]}
grid_search=GridSearchCV(class_svm2,params,n_jobs=-1,cv=5,verbose=1)
grid_search.fit(x_class_train_std,y_class_train)
optimum=grid_search.best_estimator_
print(confusion_matrix(y_class_test,optimum.predict(x_class_test_std)))
print(accuracy_score(y_class_test,optimum.predict(x_class_test_std)))
'''
