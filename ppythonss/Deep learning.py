#Deep Learning
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib. pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import*#accuracy_score,r2_score,confusion_matrix
os.chdir(r'A:\Data science\datasets for ml deep learning')
pd.set_option('display.max_columns', None)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
'''
#perceptron
from sklearn.datasets import load_iris
iris=load_iris()
#print(iris)
#data set has sepal length,sepal width, petal length,petal width
#use classification as setosa or not using perceptron
#0 is for setosa, 1 vermicolor, 2 for virginica
x=iris.data[:,(2,3)]
#print(x)# has petal length and width as numbering starts from 1 and 2
#for y use as output of setosa or not
#print(iris.target)#predicts the classes
y=(iris.target==0).astype(np.int)#type changed to int
print(y)
#Using perceptron
from sklearn.linear_model import Perceptron
perceptron_obj=Perceptron(random_state=42)
perceptron_obj.fit(x,y)#no train test data here instead we use a data set consisting of 2 columns and output using y variable
y_pred_perceptron=perceptron_obj.predict(x)
print('accuracy score is : ')
print(accuracy_score(y,y_pred_perceptron))
print('coefficient  (estimates) are: ')
print(perceptron_obj.coef_)
print('Intercept is :')
print(perceptron_obj.intercept_)
'''
import tensorflow
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
plt.imshow(X_train_full[10])
plt.show()