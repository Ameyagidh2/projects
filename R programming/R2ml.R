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
