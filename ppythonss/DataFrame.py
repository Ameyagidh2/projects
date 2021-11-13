import os
os.chdir(r"A:\Data science")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv
import math
import PIL
import IPython
import cv2
from IPython.display import display
from PIL import Image
import re
#q=np.genfromtxt("wine.csv",delimiter=";",skip_header=1)
#print(q)
#print(q[:,-1].mean())

#r=np.genfromtxt("admission.csv",dtype=None,delimiter=",",skip_header=1,names=("Serial No","GRE Score","TOEFL Score",'University Rating',"SOP","LOR","CGPA","Research","Chance of Admit"))
#print(r)
#a=r["CGPA"]
#print(a)
#r["CGPA"]=r["CGPA"] / 10 *4
#b=r["CGPA"][:20]
#print(b)
#print(r[r["Chance_of_Admit"]>0.8]["GRE_Score"].mean())
#print(r[r["Chance_of_Admit"]>0.8]["TOEFL_Score"].mean())
#print(r[r["Chance_of_Admit"]>0.8]["GRE_Score"].mean())
#print(r[r["Chance_of_Admit"]>0.8]["CGPA"].mean())

a="Ameya is a boy. He studies Mechanical engineering. Ameya is interested in computer science as well.eeb"
#print(re.findall("Ameya",a))
#print(re.search("studies",a))
#print(re.findall("Am",a))
#print(re.findall("e{2}",a))
#print(len(re.findall("^Am",a)))

#with open("ferpa.txt") as file:
    #wiki = file.read()
#print(wiki)
#p=re.findall("[\w ]*\[edit\]",wiki)
#print(p)
#print(" ")
#for x in re.findall("[\w ]*\[edit\]",wiki):
    #print(re.split("[ \[  ]",x)[0])
#for item in re.finditer("([\w ]*)(\[edit\])",wiki):
    #print(item.group(1))


#a='I refer to https://google.com and I never refer http://www.baidu.com if I have to search anything'
#print(a)
#b=re.findall("(?<=[https]:\/\/)([A-Za-z0-9.]*)",a)
#print(b)
#with open("quiz.txt") as file:
    #wiki = file.read()
#pattern='\(.\)'
#print((re.findall(pattern,wiki)))
