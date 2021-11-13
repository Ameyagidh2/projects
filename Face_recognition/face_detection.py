import cv2.cv2 as cv
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
#Knn algorithm
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


data_dir = './face_data/'
cap = cv.VideoCapture(0)
face_data = []
names = {}
class_id = 0
labels = []
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
for fx in os.listdir(data_dir):
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]
        data_i = np.load(data_dir + fx)
        #load data to np array
        face_data.append(data_i)
        # face_data contains all array values from the given path
        target = class_id * np.ones((data_i.shape[0],))
        class_id +=1
        labels.append(target)
        # storing individual data of person as faces among a group
'''
# creatung data to stor into the model
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1, 1))
# reshape states that -1,1 so the np decide the output of rows and columns keeping net as same
train_set = np.concatenate((face_data,face_labels),axis = 1)
# axis = 0 row and axis = 1 column
'''
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

train_set = np.concatenate((face_dataset, face_labels), axis=1)




font = cv.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    for face in faces:
        x,y,w,h = face
        offset = 5
        face_selection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection = cv.resize(face_selection,(100,100))
        out = knn(train_set,face_selection.flatten())
        # flatten to get 1 dimension output
        # Drawing a rectangle
        cv.putText(frame,names[int(out)],(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow("faces",frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
