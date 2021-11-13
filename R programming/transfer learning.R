#transfer learning
library('keras')
base_dir<-'A:/Data science/datasets for ml deep learning/Cat dog classifier/'
train_dir=file.path(base_dir,'train')
test_dir=file.path(base_dir,'test')
validation_dir=file.path(base_dir,'validation')
train_datagen=image_data_generator(rescale=1./255,horizontal_flip=TRUE,height_shift_range = 0.2,width_shift_range = 0.2,rotation_range = 40,shear_range = 0.2,zoom_range = 0.2
                                  ,train_dir,fill_mode="nearest"
                                  )
test_datagen=image_data_generator(rescale=1./255
                                  )

train_generator=flow_images_from_directory(
 train_dir,
 train_datagen,
 batch_size=50,
 target_size=c(150,150),
 class_mode = 'binary'
)
validation_generator=flow_images_from_directory(
  validation_dir,
  test_datagen,
  batch_size=50,
  target_size=c(150,150),
  class_mode = 'binary'
)

test_generator=flow_images_from_directory(
  test_dir,
  test_datagen,
  batch_size=50,
  target_size=c(150,150),
  class_mode = 'binary'
)


conv_base<-application_vgg16(input_shape = c(150,150,3),include_top=FALSE,weights='imagenet')

model<-keras_model_sequential()
model%>%
  conv_base%>%
  layer_flatten()%>%
  layer_dense(units=20,activation='relu')
  
model  
freeze_weights(conv_base)#trains model on features already indentified by vgg16 model  
model%>%compile(loss='binary_crossentropy',optimizer=optimizer_rmsprop(lr=2e-5),metrics=c('accuracy'))
model_history=model%>%fit_generator(train_generator,steps_per_epoch = 100,epochs=30,validation_data = validation_generator,validation_steps = 50)

evaluate<-model%>%evaluate_generator(test_generator,steps=50)
