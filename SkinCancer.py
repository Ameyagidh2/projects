#!/usr/bin/env python
# coding: utf-8

# ## Skin Cancer Detection ##
# 

# In[30]:


# every ixel needs a neuron
#  We need 3 types of neuron in last layer to classify 3 types of concer
# hidden layers to recognize the hidden patterns in the image
# Activation function to decide which neuron is active
# hidden layers recognize the parts of image
# pooling to reduce size of image matrix to half
# SSD mobile net as the model imported
# 4/1AYBe-g5odNxkJhWv800_dOmIDnJkzd-m9nWwZetZ7Rk4YNBEXLvgLg58cG4


# In[31]:


# Importing necessary Libraries
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


# In[32]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from glob import glob


# In[33]:


#skin_df = pd.read_csv("/content/drive/My Drive/MLandAI/cancer/HAM10000_metadata.csv")
skin_df = pd.read_csv(r"D:/A drive/python/Skin Cancer detection/HM001/HAM10000_metadata.csv")
skin_df


# In[34]:


image_path = {os.path.splitext(os.path.basename(x))[0]:x
                                for x in glob(os.path.join('D:/A drive/python/Skin Cancer detection/HM001/','*',"*.jpg"))}
print(image_path)


# In[35]:


for x in glob(os.path.join('D:/A drive/python/Skin Cancer detection/HM001/',"*","*.jpg")):
    print(x)
    y =  os.path.splitext(os.path.basename(x))[0] # basename is the last word before / 
    print(y)
# y has the image name    
    


# In[36]:


# Adding path of image as a column to the dataframe
skin_df2 = pd.read_csv(r"D:/A drive/python/Skin Cancer detection/HM001/HAM10000_metadata.csv")
skin_df2["path"] = skin_df2["image_id"].map(image_path.get)
skin_df2


# In[40]:


# Resize each image to 32 X 32
skin_df2['image'] = skin_df2['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))
skin_df2


# In[41]:


print(skin_df2['dx'].value_counts())


# In[47]:


# n_samples = 5
# fig, m_axs = plt.subplots(7, n_samples,figsize = (4*n_samples,3*7))
# for n_axs,(type_name,type_rows) in zip(m_axs,skin_df2.sort_values(['dx']).groupby('dx')):
#     n_axs[0].set_title(type_name)
#     for c_ax,(_,c_row) in zip(n_axs,type_rows.sample(n_samples(n_samples,random_state = 0))):
#         c_ax.imshow(c_row['image'])
#         c_ax.axis('off')


# In[79]:


# Better Way to categorize sort image data into sub folders then predict using keras
import pandas as pd
import os
import shutil


# In[80]:


# put all images in 1 folder called all_data
data_dir = "D:/A drive/python/Skin Cancer detection/HM001/all_data/"

# organize all images in new folder
des_dir = "D:/A drive/python/Skin Cancer detection/HM001/reorganized/"


# In[81]:


skin_df2 =  pd.read_csv(r"D:/A drive/python/Skin Cancer detection/HM001/HAM10000_metadata.csv")
skin_df2


# In[82]:


print(skin_df2['dx'].value_counts())


# In[83]:


# extract all the unique values to a list
labels = skin_df2['dx'].unique().tolist()
labels
# labels has all the unique labels in the list


# In[87]:


label_images = []

for i in labels:
    # create a directory of the files
    # os.mkdir(des_dir + str(i) + "/")
    sample = skin_df2[skin_df2["dx"] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir +"/"+id+".jpg"),(des_dir + i +"/"+id+".jpg") )
    label_images = []    


# In[106]:


for i in labels:
    # create a directory of the files
    # os.mkdir(des_dir + str(i) + "/")
    sample = skin_df2[skin_df2["dx"] == i]['image_id']
    label_images.extend(sample)
    print(sample)
    print(label_images)


# In[97]:


from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt


# In[98]:


# Data Augmentation
datagen = ImageDataGenerator()

train_dir = "D:/A drive/python/Skin Cancer detection/HM001/reorganized"


# In[102]:


# training keras model
train_data_keras = datagen.flow_from_directory(directory = train_dir,
                                              class_mode= "categorical", # as many types pof cancer present
                                              batch_size = 16, target_size = (32,32))
# resizing the images to 32 X 32 size


# In[104]:


x,y = next(train_data_keras)
print(x,y)
# x,y have all the numpy array values of the images


# In[105]:


# plotting the keras images
# plotting first 15 images
for i in range(0,15):
    image = x[i].astype(int)
    plt.imshow(image)
    plt.show()


# In[109]:


# Use Gan to find the best model
import seaborn as sns
import keras
from keras.utils.np_utils import to_categorical # As multiclass classification
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder # for converting the 7 labels to 1,2,3...7

np.random.seed(42)


# In[110]:


skin_df2


# In[111]:


SIZE = 32

# create a new column of labels 1 to 7 for the classes using label encoder

le = LabelEncoder()
le.fit(skin_df["dx"])

# creating a new column with label values
skin_df2["label"] = le.transform(skin_df2["dx"])

skin_df2.sample(10)


# In[116]:


# plot data to check the variations among the data values

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(221)
skin_df2["dx"].value_counts().plot(kind = "bar", ax = ax1)
ax1.set_ylabel('Count', size = 15)
ax1.set_title('dx')

ax2 = fig.add_subplot(222)
skin_df2["sex"].value_counts().plot(kind = "bar", ax = ax2)
ax2.set_ylabel('Count', size = 15)
ax2.set_title('sex')

ax3 = fig.add_subplot(223)
skin_df2["localization"].value_counts().plot(kind = "bar", ax = ax3)
ax3.set_ylabel('Count', size = 15)
ax3.set_title('localization')

ax4 = fig.add_subplot(224)
skin_df2["age"].value_counts().plot(kind = "bar", ax = ax4)
ax4.set_ylabel('Count', size = 15)
ax4.set_title('age')

plt.tight_layout()
plt.show()
# Not well balanced data


# In[119]:


# For an unbalanced data set we can now balance it to around 500 samples each then add more images duplicates to make
# number of samples per set around 500

from sklearn.utils import resample

# remove each and every class augment then concatinate it
df_0 = skin_df2[skin_df2['label'] == 0]
df_1 = skin_df2[skin_df2['label'] == 1]
df_2 = skin_df2[skin_df2['label'] == 2]
df_3 = skin_df2[skin_df2['label'] == 3]
df_4 = skin_df2[skin_df2['label'] == 4]
df_5 = skin_df2[skin_df2['label'] == 5]
df_6 = skin_df2[skin_df2['label'] == 6]

n_samples = 500

df_0_balanced = resample(df_0,replace = True, n_samples = n_samples, random_state = 42)
df_1_balanced = resample(df_1,replace = True, n_samples = n_samples, random_state = 42)
df_2_balanced = resample(df_2,replace = True, n_samples = n_samples, random_state = 42)
df_3_balanced = resample(df_3,replace = True, n_samples = n_samples, random_state = 42)
df_4_balanced = resample(df_4,replace = True, n_samples = n_samples, random_state = 42)
df_5_balanced = resample(df_5,replace = True, n_samples = n_samples, random_state = 42)
df_6_balanced = resample(df_6,replace = True, n_samples = n_samples, random_state = 42)

# combine the  dataframes together

skin_df_balanced = pd.concat([df_0_balanced,df_1_balanced,df_2_balanced,df_3_balanced,df_4_balanced,df_5_balanced,df_6_balanced])

skin_df_balanced


# In[121]:


skin_df_balanced['dx'].value_counts()


# In[122]:


# Resizing the balanced data set
image_path2 = {os.path.splitext(os.path.basename(x))[0]:x
                                for x in glob(os.path.join('D:/A drive/python/Skin Cancer detection/HM001/','*',"*.jpg"))}
print(image_path2)

skin_df_balanced["path"] = skin_df_balanced["image_id"].map(image_path2.get)

# Resize each image to 32 X 32
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))


# In[123]:


skin_df_balanced


# In[124]:


# x is the image and y is the label
x = np.asarray(skin_df_balanced["image"].tolist()) # convert to list to pass into a numpy array


# In[125]:


x


# In[128]:


X = x /255 # normalize the values between 0 to 1
# y is the independent label output
y = skin_df_balanced['label']
y


# In[129]:


y_cat = to_categorical(y,num_classes = 7)# makes the categorival output here for deep learning
y_cat


# In[135]:


# training and testing data split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y_cat, test_size= 0.25,random_state=42)
# Auto keras to get the best hyper parameters for the model
# Model defination
num_classes = 7


# model creation
model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])


# In[136]:


# Training

batch_size = 16
epochs = 50
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)


# In[138]:


score = model.evaluate(x_test,y_test)
print('Test accuracy', score[1])


# In[144]:


loss_curve = history.history['loss']
epoches = range(1,len(loss) + 1)
plt.plot(epochs,loss_curve,'y',label = 'loss')
plt.show()


# In[145]:


val_loss_curve = history.history['val_loss']
epoches = range(1,len(loss) + 1)
plt.plot(epochs,val_loss_curve,'y',label = 'validation loss')
plt.show()


# In[147]:


acc = history.history['acc']
epoches = range(1,len(loss) + 1)
plt.plot(epochs,acc,'y',label = 'Accuracy')
plt.show()


# In[142]:


# predict
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis = 1)
y_true= np.argmax(y_test,axis = 1) 

y_pred


# In[ ]:




