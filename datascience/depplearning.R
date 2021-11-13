library('dummies')
library('caTools')
#install.packages('keras')
library(keras)
#install_keras()


#Classification Model
fashion_mnist<-dataset_fashion_mnist()
#train test split
c(train_images,train_labels)%<-%fashion_mnist$train
c(test_images,test_labels)%<-%fashion_mnist$test

#plotting data
fobject<-train_images[5,,]#plotting 5th image
plot(as.raster(fobject,max=255))

#Predicting the class name of the image
class_names=c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
              )

class_names[train_labels[5]+1]

#normalizing the data  X
train_images<-train_images/255
test_images<-test_images/255
dim(train_images)#3 D
dim(train_labels)#1 D
str(train_images)
#spilt train set into validation set
valid_indices<-1:5000
val_images<-train_images[valid_indices,,]#3 pixels 2d 28x28
part_train_images<-train_images[-valid_indices,,]

val_labels<-train_labels[valid_indices]#1d
part_train_labels<-train_labels[-valid_indices]

#making the keras model of layers
model<-keras_model_sequential()
model%>%
  layer_flatten(input_shape = c(28,28))%>%
  layer_dense(units=128,activation = 'relu')%>%
  layer_dense(units=10,activation = 'softmax')

#model compilation
model%>%compile(
  loss='sparse_categorical_crossentropy',
  optimizer='sgd',
  metrics=c('accuracy')
  )

model%>%fit(part_train_images,part_train_labels,validation_data=list(val_images,val_labels),batch_size=100,epochs=30)

#predicting the values output
score<-model%>%evaluate(test_images,test_labels)
#predicting the class
cat('test loss',score$loss,'\n')
cat('test accuracy',score$acc,'\n')
a=2
plot(as.raster(test_images[a,,],max=1))#2nd test image
class_predict<-model%>%predict_classes(test_images)
#output of 2nd test image
class_predict[a]+1# class corresponding to the first 20 data images and class numbering starting with 0 so use +1
#output alternate way
prediction<-model%>%predict(test_images)
class_names[which.max(prediction[a,])]

#Neural net instead of keras for simple neural networks
install.packages('neuralnet')
require(neuralnet)

hours=c(20,10,30,20,50,30)
marks=c(90,20,20,10,50,80)
passed<-c(1,0,0,0,1,1)
data_college=data.frame(hours,marks,passed)
?neuralnet
nn=neuralnet(passed~marks+hours,data=data_college,hidden=c(3,2),act.fct='logistic',linear.output=FALSE)#if linear output false then classification
plot(nn)
thours=c(60,20,30)
tmarks=c(80,30,20)
test_data=data.frame(thours,tmarks)
predict_test<-compute(nn,test_data)#
prob<-predict_test$net.result
prob
predicted_class<-ifelse(prob>0.5,1,0)
predicted_class


#Regression model using functional api method
library(keras)
boston_housing<-dataset_boston_housing()
#split
c(train_data,train_labels)%<-%boston_housing$train
c(test_data,test_labels)%<-%boston_housing$test
#normalize the data not just dividing by 255 but using x-mean/sd formula which the software does for us
train_data=scale(train_data)
#now we only have train data thats why we use this scale to even transform our test data
cal_means <- attr(train_data,"scaled:center")
call_std<- attr(train_data,'scaled:scale')
test_data=scale(test_data,center=cal_means,scale=call_std)
dim(train_data)
#Using functional api for complex and regression problems 
input_layer<-layer_input(shape=dim(train_data)[2])#columns in train data is same as variables

predictions<-input_layer%>%
  layer_dense(units=64,activation='relu')%>%
  layer_dense(units=64,activation='relu')%>%
  layer_dense(units=1)
  

model<-keras_model(inputs=input_layer,output=predictions)
model%>%compile(optimizer='rmsprop',
                loss='mse',
                metrics=list('mean_absolute_error')
                )
model %>% fit(train_data, train_labels, epochs = 20, batch_size=100)
score<-model%>%evaluate(test_data,test_labels)
summary(model)
#for complex networks use a concatinate layer
input_layer<-layer_input(shape=dim(train_data)[2])
predictions_func<-input_layer%>%
  layer_dense(units=64,activation='relu')%>%
  layer_dense(units=64,activation='relu')

main_output<-layer_concatenate(c(predictions_func,input_layer))
model_func<-keras_model(inputs=input_layer,output=main_output)

model_func%>%compile(optimizer='rmsprop',
                     loss='mse',
                     metrics=list('mean_absolute_error')
                     )
model_func%>%fit(train_data,train_labels,epochs=20,batch_size=100)
score_fumc<-model_func%>%evaluate(test_data,test_labels)
summary(model_func)
#model_func is complex functional api


#CNN Model
library(keras)
fashion_mnist<-dataset_fashion_mnist()
#split
c(train_images,train_labels)%<-% fashion_mnist$train
c(test_images,test_labels)%<-%fashion_mnist$test
#Normalize data
train_images<-train_images/255
test_images<-test_images/255
dim(train_images)

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')
#Validation set split
valid_indices<-1:5000
valid_images<-train_images[valid_indices,,]
valid_labels<-train_labels[valid_indices]
part_train_images<-train_images[-valid_indices,,]
part_train_labels<-train_labels[5001:60000]

#Reshape data for cnn model
dim(part_train_images)#55000
dim(test_images)#10000 
dim(valid_images)#5000 
part_train_images<-array_reshape(part_train_images,c(55000,28,28,1))#1 for black and white channels and 3 for rgb channels
test_images<-array_reshape(test_images,c(10000,28,28,1))
valid_images<-array_reshape(valid_images,c(5000,28,28,1))

#Model cnn layers and then regular ann layers

model1<-keras_model_sequential()%>%
 layer_conv_2d(strides = 2,filters = 32,kernel_size =c(3,3),input_shape = c(28,28,1),activation='relu')%>%
 layer_max_pooling_2d(pool_size=c(2,2))

model<-model1%>%
  layer_flatten() %>%
  layer_dense(units=300,activation='relu')%>%
  layer_dense(units=100,activation='relu')%>%
  layer_dense(units=10,activation = 'softmax')

model%>%compile(
  loss='sparse_categorical_crossentropy',#for non overlapping more than 2 classes in model
  metrics=('accuracy'),
  optimizer='sgd'
)

model%>%fit(part_train_images,part_train_labels,validation_data=list(valid_images,valid_labels),epochs=20,batch_size=64)

cnn_score<-model%>%evaluate(test_images,test_labels)
class_pred<-model%>%predict_classes(test_images)
#model prediction
class_pred[1:20]
class_names[class_pred[1:20]]
#actual values
class_names[test_labels[1:20]]



