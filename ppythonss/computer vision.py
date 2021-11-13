import pandas as pd
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#blending images together
import cv2
import numpy as np
apple=cv2.imread('A:\\python\\MATLAB\\apple.jpg')
orange=cv2.imread('A:\\python\\MATLAB\\org.jpg')
apple=cv2.resize(apple,(512,512))
orange=cv2.resize(orange,(512,512))
#built in function to merge 2 images take their parts and then merge
'''
orange=cv2.imread('A:\python\MATLAB\orange.jpg')
print(orange.shape)
apple=cv2.imread('A:\python\MATLAB\apple.jpg')
apple=cv2.resize(apple,(512,512))
apple=cv2.resize(apple,(512,512))
#apple_orange_combined=np.hstack((apple[:,:256],orange[:,256:]))#this is equivalent to size of image
apple_orange_combined=np.hstack((apple[:,:112],orange[:,112:]))
#cv2.imshow('apple',apple)
#cv2.imshow('orange',orange)
cv2.imshow('apple_orange_merged',apple_orange_combined)
cv2.waitKey()
cv2.destroyAllWindows()
'''


#combined_image
#using blend form of image reduce the image to lowest form then blend
#laplacian(orange)
def laplacian(image):
    gp=[image]
    for i in range(3):
      image=cv2.pyrDown(image)
      gp.append(image)
    lp=[gp[3]]
    #Laplacian we take difference between between last and the one previous
    for i in range(3,0,-1) :
      gauss_expand=cv2.pyrUp(gp[i])
      lap=cv2.subtract(gp[i-1],gauss_expand)
      lp.append(lap)
      #cv2.imshow(str(i),lap)
      cv2.waitKey()
      cv2.destroyAllWindows()
    return lp


#decomposing the image into smallest form so it is easy to the built them up by laplacian methods as small form easy to blend
lp_apple=[]
lp_orange=[]
lp_apple=laplacian(apple)
lp_orange=laplacian(orange)

stack=[]
#joining 2 objects together
#orange=cv2.resize(orange,(512,512))
for apple_lap,orange_lap in zip(lp_apple,lp_orange):
    col,rows,ch=apple_lap.shape
    lap=np.hstack((apple_lap[:,:int(col/2)],orange_lap[:,int(col/2):]))
    stack.append(lap)

recons=stack[0]
for i in range(1,3):
    recons=cv2.pyrUp(recons)
    recons=cv2.add(stack[i],recons)


cv2.imshow('images blended',recons)
#lp_apple=laplacian(mango)
#cv2.imshow('mango',mango)
#cv2.imshow('orange',orange)
#cv2.imshow()
cv2.waitKey()
cv2.destroyAllWindows()
