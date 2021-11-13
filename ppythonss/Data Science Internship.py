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
a=["rice","apple","chocolates"]
b=["mon","tue","wed"]
c=pd.Series(a,index=b)
print(c)
print(type(c))
d={"a":["rice","apple","chocolates"],
"b":["mon","tue","wed"]}
df=pd.DataFrame(d,columns=["a","b"])
print(df)
df.to_csv("trial.csv")
import pandas as pd
cars=pd.read_csv("mtcars.csv")
cars
cars.info()
cars=pd.read_csv("mtcars.csv", index_col="model")
cars.sort_index(ascending=True,inplace=True)
cars
cars['hp'].mean()
cars.loc["Honda Civic",'mpg':'hp']
cars.iloc[11]
import matplotlib.pyplot as plt
x=[1,10]
y=[2,19]
plt.plot(x,y,c="b",marker="D",linestyle=':')
plt.xlabel("X-label")
plt.ylabel("Y-label")
plt.title("Lineplot")
plt.show()

import matplotlib.pyplot as plt
x=[1,10]
y=[2,19]
plt.scatter(x,y,c="b",marker="D",linestyle=':',s=100)
plt.xlabel("X-label")
plt.ylabel("Y-label")
plt.title("Time pass")
plt.show()
import pandas as pd
df=pd.read_csv("/content/Titanic (1).csv")
print(df)


