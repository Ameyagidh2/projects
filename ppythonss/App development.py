import os
os.chdir(r"A:\App development\Jumpbot")
'''
# first GUI
window=tkinter.Tk()
window.geometry("800x300")

def text1():
    a=dollars.get()
    print("hi")
    print(a)
    inr = (float(dollars.get()) * 75)
    t1.delete("1.0",tkinter.END)
    t1.insert(tkinter.END," HI Welcome to the dollar to inr converter \n")
    t1.insert(tkinter.END, a)
    t1.insert(tkinter.END," $ in inr is ")
    t1.insert(tkinter.END,inr)
    t2.insert(tkinter.END,"See ya!")



b1=tkinter.Button(window,text="Sumbit",command=text1,width=30,height=2)
#b1.pack()  #similar to b1.grid just you cannot control its position
b1.grid(row=0,column=1)
0
dollars=tkinter.StringVar()
e1=tkinter.Entry(window,textvariable=dollars,width=30,justify="center")
e1.grid(row=1,column=1)

t1=tkinter.Text(window,width=50,height=2)
t1.grid(row=1,column=3)

t2=tkinter.Text(width=50,height=2)
t2.grid(row=1,column=5)

window.mainloop()
'''

'''
mylist = ["apple", "banana", "cherry"]
a=random.sample(mylist,len(mylist))

print(a)
'''
'''
# Jumble bot
import tkinter
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
'''
'''
# Pingball game
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.vector import Vector
from kivy.properties import NumericProperty,ReferenceListProperty,ObjectProperty
from kivy.clock import Clock
from random import randint

class PongPaddle(Widget):
     score=NumericProperty(0)
     def bounce_ball(self,ball):
      if self.collide_widget(ball):
          ball.v_x *=-1


class PongBall(Widget):
    v_x=NumericProperty(0)
    v_y=NumericProperty(0)
    vnet=ReferenceListProperty(v_x,v_y)
    def move(self):
        self.pos=Vector(*self.vnet)+self.pos
class PongGame(Widget):
   # PongPaddle=ObjectProperty(None)
    ball=ObjectProperty(None)
    p1=ObjectProperty(None)
    p2 = ObjectProperty(None)
    def serve(self):
        self.ball.vnet=Vector(5,0).rotate(randint(0,360))
    def update(self,dt):
        self.ball.move()
        if (self.ball.y<0 )or (self.ball.y>self.height-50):
            self.ball.v_y*=-1
        if (self.ball.x<0 ):
            self.ball.v_x *= -1
            self.p2.score+=1
        if  (self.ball.x>self.width-50):
            self.ball.v_x*=-1
            self.p1.score += 1
        self.p1.bounce_ball(self.ball)
        self.p2.bounce_ball(self.ball)

    def on_touch_move(self,touch):
        if touch.x<self.width/1/4:
            self.p1.center_y=touch.y
        if touch.x>self.width*3/4:
            self.p2.center_y=touch.y
class PongApp(App):

    def build(self):
        game = PongGame()
        game.serve()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game

PongApp().run()
'''
'''
# password generator
tup = (('A','/|'),('B',"6"),("a","@"),('D','|?'),('S','$'),('L','1_-'))

def password_secure(password):
    for a,b in tup:
      password=password.replace(a,b)
    return password

password=input("Enter your password: ")
decision=input("Lower allowed (Y/N) ")
if (decision=="Y")or(decision=="y"):
    print(f"Encrypted password is: {password_secure(password)} ")
else:
    print(f"Encrypted password is: {password_secure(password).lower()} ")
'''