install.packages("caret")
install.packages("fansi")
install.packages("stringi")
install.packages(
  "ggplot2",
  repos = c("http://rstudio.org/_packages",
            "http://cran.rstudio.com")
)
install.packages("ggplot2", dependencies = TRUE)
install.packages("caret", dependencies = TRUE) 
library(ggplot2)
library(caret)
ibrary('rpart')
library('rpart.plot')
df<-read.csv("House_Price.csv",header=T)
df
#univariate edd analysis
hist(df$crime_rate)
summary(df)
#skewed graph plot
pairs(~price+n_hot_rooms+rainfall+crime_rate,data=df)#get scatter plots for price vs all other inputs
#skewed for crime rates and a relationship exists for rainfall and crime rates n hot rooms is skewed at 1 side
barplot(table(df$bus_ter))
#not useful for analysis variable
# removing outliers found in n_hot_rooms and rainfall
ul<-quantile(df$n_hot_rooms,0.99)
ul
lv<-quantile(df$rainfall,0.01)
lv
#replacing now
d1f$n_hot_rooms[df$n_hot_rooms>3*ul]<-3*ul
summary(df)
df1$rainfall[df$rainfall<0.3*lv]<-0.3*lv
summary(df)
#min changed from 3 to 6
#filling missing values
mean(df$n_hos_beds,na.rm=TRUE)
which(is.na(df$n_hos_beds))
df$n_hos_beds[which(is.na(df$n_hos_beds))]<-mean(df$n_hos_beds,na.rm=TRUE)
summary(df$n_hos_beds)
# for checking if command worked which(is.na(df$n_hos_beds))
df

#Bivariate variable transformation
pairs(~df$price+df$crime_rate)
#transform to log and add 1 to prevent problem due to log function
df$crime_rate<-log(1+df$crime_rate)
df$avg_dist<-(df$dist1+df$dist2+df$dist3+df$dist4)/4
df<-df[,-7:-10]
df<-df[,-14]
df1<-df
#dummy variable creation for changing categorial variables to 0 and 1 for regression analysiis
#install.packages('dummies')
library('dummies')
df<-dummy.data.frame(df)
df1<-dummy.data.frame(df1)
df1<-df

df<-df[,-9]
df<-df[,-14]
#correlation 
round(cor(df),2)
#parks and air quality high correlation hence remove 1 which is not having higher coefficient with our dependent variable price 
df<-df[,-16]
df1<-df1[,-16]
df2<-df1
#rainfall and n_hot_beds not very imporant as low r with x dependent variable
df1<-df
#Linear regression rsquare is how close data fitted wrt regression line,std error spread of data Relational standard error RSE is u]the absolute distance difference lsm method of the points from the regression line., t probability gives if low the ha that is anti hypothesis value is small then good ie p{<t}should be close to 0 and estimates are the intercetps b0 and slope B1 
simple_model<-lm(price~room_num,data=df)#Multi regression 
summary(simple_model)
#scatter plot
plot(df$room_num,df$price)
abline(simple_model)#Regression line plotted
#Multilinear regression
multi_reg<-lm(price~.,data=df)
summary(multi_reg)

#Train test split
#install.packages('caTools')
library('caTools')
set.seed(0)#random_state=0
split=sample.split(df,SplitRatio=0.8)
rm(train_dataset)
train_dataset=subset(df,split==TRUE)
test_dataset=subset(df,split==FALSE)
lm_a=lm(price~.,data=train_dataset)#Multi regression on training data
train_pred=predict(lm_a,train_dataset)
test_pred=predict(lm_a,test_dataset)
mean((train_dataset$price-train_pred)^2)
mean((test_dataset$price-test_pred)^2)
#Different models like subsets,forward and backward propogation
#Subsets
#install.packages("leaps")
library("leaps")
lm_a=regsubsets(price~.,data=df,nvmax=15)
which.max(summary(lm_a)$adjr2)
summary(lm_a)$adjr2

#Forward
lm_f=regsubsets(price~.,data=df,nvmax=15,method="forward")
which.max(summary(lm_f)$adjr2)
summary(lm_f)$adjr2

#Ridge and Lasso regression
#install.packages('glmnet')
library("glmnet")
rm(y_rid)
x_rid1=model.matrix(price~.,data=df)[,-1]#To standardize x and last column of dataset dropped
y_rid1=df$price
#alpha values generation
grid=10^seq(10,-2,length=100)
print(grid)
#fitting alpha by grid values
#?glmnet
lm_ridge_reg=glmnet(x_rid1,y_rid1,alpha=0,lambda=grid)#alpha=0 for ridge
summary(lm_ridge_reg)#vague values not r^2 value
#finding r^2 value
cv_fit=cv.glmnet(x_rid1,y_rid1,alpha=0,lambda=grid)
plot(cv_fit)
#finding where is the lowest point in the fit
opt_lambda=cv_fit$lambda.min
print(opt_lambda)
tss=sum((y_rid1-mean(y_rid1))^2)
#?predict
y_rid_pred=predict(lm_ridge_reg,s=opt_lambda,newx=x_rid1)
rss=sum((y_rid_pred-y_rid1)^2)
r2square=1-rss/tss
r2square
rm(df2)

#EDD
df2<-read.csv("House-Price1.csv",header=T)
boxplot(df2$n_hot_rooms)
pairs(~df2$price+df$rainfall)
barplot(table(df$airport))
ul<-quantile(df2$n_hot_rooms,0.99)
ul
lv<-quantile(df2$rainfall,0.01)
lv
#replacing now
df2$n_hot_rooms[df2$n_hot_rooms>3*ul]<-3*ul
summary(df2)
df2$rainfall[df2$rainfall<0.3*lv]<-0.3*lv
summary(df2)
df2$n_hos_beds[which(is.na(df2$n_hos_beds))]<-mean(df2$n_hos_beds,na.rm=TRUE)
summary(df2$n_hos_beds)
df3<-df2
df3<-df2[,-6:-9]
df3<-df3[,-13]
df3<-dummy.data.frame(df3)
df3<-df3[,-14]
df3<-df3[,-8]
df3$avg_dist<-df1$avg_dist
df2<-df3

#Logistic Regression
#Single Variable
library('glmnet')
glm.fit=glm(Sold~price,data=df2,family=binomial)
summary(glm.fit)

#Multiple Variable
glm.fit=glm(Sold~.,data=df2,family=binomial)
summary(glm.fit)
glm.probs=predict(glm.fit,type='response')
#Summary of percentage of first 10 valuesg
glm.probs[1:10]
#Grouping into classes first make and table of all 506 nos then give condition to add a yes with threshold of 0.5
glm.pred=rep('No',506)
glm.pred[glm.probs>0.5]<-'YES'
#Confusion Matrix
table(glm.pred,df3$Sold)

# LDA Linear Discriminant Analysis
#install.packages('MASS')
library("MASS")
lda_fit = lda(Sold~.,data=df2)
y_predicted=predict(lda_fit,df2)
print(y_predicted)
#y_predicted$posterior
lda_y.class=y_predicted$class
table(lda_y.class,df2$Sold)
df<-df2



# Perfect classification model
library('caTools')
?sample.split()
set.seed(0)
split=sample.split(df, SplitRatio = 0.8)

training_dataset=subset(df,split==TRUE)
testing_dataset=subset(df,split==FALSE)
logreg_fitted=glm(Sold~.,data=training_dataset,family=binomial)
?predict
y_predicted_probs=predict(logreg_fitted,testing_dataset,type='response')
y_predicted_value=rep("NO",120)
y_predicted_value[y_predicted_probs>=0.5]="Yes"
print("confusion matrix using Logistic regression")
table(testing_dataset$Sold,y_predicted_value)


#Perfect LDA
lda_fitted=lda(Sold~.,data=training_dataset)
lda_predicted=predict(lda_fitted,testing_dataset)
lda_predicted.class=lda_predicted$class #for lda form classes
lda_predicted.class
print("confusion matrix using LDA")
table(testing_dataset$Sold,lda_predicted.class)


#KNN K NEAREST NEIGHBOORS
#install.packages("class")
library('class')
x_train=training_dataset[,-16]
y_train=training_dataset$Sold
x_test=testing_dataset[,-16]
y_test=testing_dataset$Sold
kn=3
x_train_s=scale(x_train)
x_test_s=scale(x_test)
set.seed(0)
summary(df)

y_pred_knn=knn(x_train_s,x_test_s,y_train,k=kn)
print("Confusion matrix Knn")
table(y_pred_knn,y_test)

multi_reg<-lm(price~.,data=df)
summary(multi_reg)

#Decision trees
movie<-read.csv('Movie_regression.csv',header=T)
summary(movie)
#Time_taken has 12 na values
movie$Time_taken[is.na(movie$Time_taken)]<-mean(movie$Time_taken,na.rm=TRUE)
summary(movie)
#get dummies
movie<-dummy.data.frame(movie)
movie<-movie[,-12]
movie<-movie[,-15]
#data split
library('caTools')
split=sample.split(movie,SplitRatio = 0.8)
train_tree=subset(movie,split==TRUE)
test_tree=subset(movie,split==FALSE)
#Forming the decision tree
regtree<-rpart(formula=Collection~.,data=train_tree,control=rpart.control(max_depth=3))
test_tree_y=predict(regtree,test_tree,type="vector")
mse=mean((test_tree_y-test_tree$Collection)^2)
mse
#Plotting descision tree
rpart.plot(regtree,box.palette = 'RdBu',digits=-3)
#Pruning a tree 
#allow the tree to grow fully
full_tree<-rpart(formula=Collection~.,data=train_tree,control=rpart.control(cp=0))
plotcp(regtree)
#plotcp(full_tree)
printcp(full_tree)
#find minimum cp value of xerror
mincp<-regtree$cptable[which.min(regtree$cptable[,'xerror']),'CP']
mincp
#making prune tree
prune_tree<-prune(full_tree,cp=mincp)
rpart.plot(prune_tree,box.palette = 'RdBu',digits=-3)
y_prunedTree_predicted=predict(prune_tree,test_tree,type='vector')
mse_tree_pruned=mean((y_prunedTree_predicted-test_tree$Collection)^2)

#Classification decision tree to predict oscar
df_clf_tree=read.csv('Movie_classification.csv',header=T)
summary(df_clf_tree)
#Time_taken has nas
df_clf_tree$Time_taken[is.na(df_clf_tree$Time_taken)]<-mean(df_clf_tree$Time_taken,na.rm=TRUE)
#dummy variables creation
df_clf_tree<-dummy.data.frame(df_clf_tree)
rm(clf_tree)
df_clf_tree<-df_clf_tree[,-15]
#split into test train data sets
split2=sample.split(df_clf_tree,SplitRatio = 0.8)
train2_tree=subset(df_clf_tree,split2==TRUE)
test2_tree=subset(df_clf_tree,split2==FALSE)

#Classification model
clf_tree<-rpart(formula=Start_Tech_Oscar~.,data=train2_tree,method='class',control=rpart.control(max_depth=3))
y_pred_clf_tree=predict(clf_tree,test2_tree,type='class')
table(test2_tree$Start_Tech_Oscar,y_pred_clf_tree)
#plot of classification tree
rpart.plot(clf_tree,box.palette='RdBu',digits=-3)

#Bagging
install.packages('randomForest')
library('randomForest')
set.seed(0)
bagging_obj=randomForest(formula=Collection~.,data=train_tree,mtry=17)
y_predicted_bagging=predict(bagging_obj,test_tree)
#table(test_tree$Collection,y_predicted_bagging)
mse_bag=mean((y_predicted_bagging-test_tree$Collection)^2)
mse_bag

#Random forest
rand_forest=randomForest(formula=Collection~.,data=train_tree,ntree=500)
y_randomforest_predicted=predict(rand_forest,test_tree)
mse_rand_forest=mean((test_tree$Collection-y_randomforest_predicted)^2)
mse_rand_forest


#Boosting  
#1  Gradient_boost  on Regression test2_tree
install.packages("gbm")
library('gbm')
?gbm
boost_clf=gbm(formula=Collection~.,data=train2_tree,distribution = "gaussian",shrinkage = 0.02,interaction.depth = 4,verbose = FALSE,n.trees = 5000)
y_pred_boost=predict(boost_clf,test2_tree,n.trees=5000)
mse_boost=mean((y_pred_boost-test2_tree$Collection)^2)
mse_boost
#Dependencies
#library('stringi')
#install.packages("stringi")
library('installr')
install.packages("caret",
                 repos = "http://cran.r-project.org", 
                 dependencies = c("Depends", "Imports", "Suggests"))
install.packages("tidyverse")
library(tidyverse)
install.packages("ggplot2")
library("ggplot2")
sessionInfo()
#2  ADA Boost on classification hence using tes2t_tree
install.packages('adabag')
library(adabag)
df_clf_tree_ada<-df_clf_tree
#convert the dependent variabke into factor for input to our model
df_clf_tree_ada$Start_Tech_Oscar<- as.factor(df_clf_tree_ada$Start_Tech_Oscar)
library('caTools')
set.seed(0)
split3=sample.split(df_clf_tree,SplitRatio = 0.8)
train3_tree=subset(df_clf_tree_ada,split3==TRUE)
test3_tree=subset(df_clf_tree_ada,split3==FALSE)
#Creating ada bosst classifier as ada on classification only possible in r programming
ada_boost<-boosting(Start_Tech_Oscar~.,data=train3_tree,mfinal=1000,boos=TRUE)
predict_ada=predict(ada_boost,test3_tree)
table(predict_ada$class,test3_tree$Start_Tech_Oscar)
#t_I<-ada_boost$trees
#plot(t_I)
#text(t_Ipretty=100)

#XG Boost
install.packages('xgboost')
library('xgboost')
library('caTools')
df_clf_xgboost=read.csv('Movie_classification.csv',header=T)
summary(df_clf_tree)
#Time_taken has nas
df_clf_xgboost$Time_taken[is.na(df_clf_xgboost$Time_taken)]<-mean(df_clf_xgboost$Time_taken,na.rm=TRUE)
set.seed(0)
split3=sample.split(df_clf_xgboost,SplitRatio = 0.8)
train3_tree=subset(df_clf_xgboost,split3==TRUE)
test3_tree=subset(df_clf_xgboost,split3==FALSE)

train_Y_xgboost=train3_tree$Start_Tech_Oscar=='1'#to make output as boolean
train_X_xgboost<-model.matrix(Start_Tech_Oscar~.-1,data=train3_tree)
train_X_xgboost<-train_X_xgboost[,-12]
rm(test__Y_xgboost)
test__Y_xgboost=train3_tree$Start_Tech_Oscar=='1'
#test__Y_xgboost
test_X_xgboost<-model.matrix(Start_Tech_Oscar~.-1,data=test3_tree)   #like get dummies function
test_X_xgboost<-test_X_xgboost[,-12]

#Seperating test and train data into matrix form as input to xgb model

x_matrix<-xgb.DMatrix(data=train_X_xgboost,label=train_Y_xgboost)#data in xgboost is in form of d matrix so make train and test d mat
x_matrix_t<-xgb.DMatrix(data=test_X_xgboost,label=test_Y_xgboost)
test_Y_xgboost
test_X_xgboost

xgbboosting<-xgboost(data=x_matrix,nround=50,objective='multi:softmax',eta=0.3,num_class=2,max_depth=100)
xgb_pred<-predict(xgbboosting,x_matrix_t)#test set
table(test_Y_xgboost,xgb_pred)

# SVM
#classification for regression just no need of as factor
install.packages('e1071')
library(e1071)
library('dummies')
library('caTools')
movie_svm=read.csv('Movie_classification.csv',header=TRUE)
summary(movie_svm)
set.seed(0)
movie_svm$Time_taken[is.na(movie$Time_taken)]<-mean(movie_svm$Time_taken,na.rm=TRUE)
#Dummy
movie_svm=dummy.data.frame(movie_svm)
#Split
split=sample.split(movie,SplitRatio = 0.8)
train_class=subset(movie_svm,split==TRUE)
test_class=subset(movie_svm,split==FALSE)
#factor classification
test_class$Start_Tech_Oscar<-as.factor(test_class$Start_Tech_Oscar)
train_class$Start_Tech_Oscar<-as.factor(train_class$Start_Tech_Oscar)
#training the model for model
svm_class=svm(Start_Tech_Oscar~.,data=train_class,kernel='linear',cross=4,scale=TRUE,cost=1)
summary(svm_class)
y_pred=predict(svm_class,test_class)
table(predict=y_pred,test_class$Start_Tech_Oscar)
#tunning
tune.out<-tune(svm,Start_Tech_Oscar~.,data=train_class,kernel="polynomial",cross=4,ranges=list(cost=c(0.001,10.01,10,100,1000,20,50),degree=c(1,2,3,4,5)))
tune.out.best=tune.out$best.model
tune.out_pred=predict(tune.out.best,test_class)
#for predicting value of radial use gamma instead of degree
tune.out.best
table(predict=tune.out_pred,true=test_class$Start_Tech_Oscar)

