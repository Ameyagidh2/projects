import os
os.chdir(r"A:\Data science")
import numpy as np
import pandas as pd
import matplotlib as plt
import cv
import math
import PIL
import IPython
import cv2
from IPython.display import display
from PIL import Image
#df=pd.read_csv("a.csv")
#print(df)


#img=cv2.imread("am.JPG")
#cv2.imshow('image',img)
#cv2.waitKey()
#cv2.destroyAllWindows()

im = Image.open('am.JPG')
display(im)
im.show()
arr=np.array(im)
#print(arr.shape)
print(arr)
#masking
mask=np.full(arr.shape,255)
print(mask)
#subtracting to form a modified array
modified=arr-mask
modified=modified*-1
print(modified)
#showing image
modified=modified.astype(np.uint8)
a=Image.fromarray(modified)
a.show()

q=np.genfromtxt("wine.csv",delimiter=";",skip_header=1)
print(q)
