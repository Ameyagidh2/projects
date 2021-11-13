import tkinter
import os
os.chdir(r"A:\App development\Jumpbot")
#from tkinter import *
from tkinter import messagebox
import random
#from random import shuffle
window=tkinter.Tk()
window.geometry("300x300")
window.configure(background="#120E43")
window.title("Ameya's Jumpbot")
window.iconbitmap("Icon.ico")
press=tkinter.StringVar()
lb1=tkinter.Label(window,font="times 20",bg="#CAD5E2",fg="#0D0D0D")
lb1.pack()
answers=["India","Africa","America","Australia","Russia"]
questions=[]
for i in answers:
    i=list(i)
    print(i)
    random.shuffle(i)
    #print(w)
    questions.append(i)
print(questions)
nums=random.randint(0,len(questions)-1)
def initial():
    global questions,answer,nums
    lb1.configure(text=questions[nums])


def submitt():
    global press,questions,answers,num
    answer2=press.get()
    if answer2 in answers:
      tkinter.messagebox.showinfo("Success","Your answer was correct")
    else:
        tkinter.messagebox.showinfo("Error","Your answer is incorrect")
        #e1.delete(0,tkinter.END)
       #t1.insert(tkinter.END,answer)
def resett():
    #e1.insert(tkinter.END,"Resttted")
    nums=random.randint(0,len(questions)-1)
    lb1.configure(text=questions[nums])
    e1.delete(0, tkinter.END)

e1=tkinter.Entry(window,fg="#0D0D0D",width=30,textvariable=press,justify="center")
e1.pack(ipady=5,ipadx=5)


b1=tkinter.Button(window,bg="#6A1B4D",width=30,height=2,text="Submit",command=submitt)
b1.pack(pady=40)

b2=tkinter.Button(window,bg="#6A1B4D",width=30,height=2,text="Reset",command=resett)
b2.pack()
#t1=tkinter.Text(window,width=30,height=2)
#t1.pack()
initial()
window.mainloop()