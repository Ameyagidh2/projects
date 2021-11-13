
#Blink
from  gpiozero import LED
from time import sleep
'''
led=LED(13)
while True:
    led.on()
    print('NO Led')
    sleep(0.5)
    led.off()
    print('OFF Led')
    sleep(0.5)
 '''
#Button
from gpiozero import Button
'''
button=Button(2)
while true:
 if button.is_pressed;
  led.on()
  print('button pressed')
 else:
   led.off()
   print('led is of) 
'''

#Led controller
from gpiozero import LEDBoard
'''
led=LED(17)
LED_BOARD=LEDBoard(17,21,22)
list1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
while True:
 for i in list1:
  led.value(i)
  sleep(0.5)
#using ledboard
while True:
 LED_BOARD.value=(0.1,0.5,1)
 sleep(0.5)
 LED_BOARD.value=(0.1,0.5,1)
 sleep(0.5)
'''
#Ultrasonic sensor
from gpiozero import DistanceSensor
'''
sensor=DistanceSensor(24,23)#echo,trigger
while True:
    print(sensor.distance*100)
'''
#Buzzer
from gpiozero import Buzzer
'''
buzzer=Buzzer()
while True:
    buzzer.on()
    sleep(0.5)
    buzzer.off()
'''
# Servo
from gpiozero import Servo
'''
gp=17
servo1=Servo(gp)
while True:
 servo.max()
 sleep(1)  
 servo.min()
 sleep(1)
 servo.max()
 sleep(1)  
 '''
#Face Detection
#import face-recognition
#import picamera
import cv2