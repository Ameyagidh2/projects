# Jarvis ChatBot AI assistant project
import pyttsx3
import datetime
import speech_recognition as sr
import  wikipedia
import smtplib
import webbrowser as wb
import psutil
import pyjokes
import os
import pyautogui
import random
import time
import json
import requests
from urllib.request import urlopen
import wolframalpha

engine = pyttsx3.init()
wolframalpha_app_id = "wolframe app id will go here"

def speak(audio):
# Text to Speech
    engine.say(audio)
    engine.runAndWait()

def take_command():
    # makes user speech into text
    r = sr.Recognizer()
    # Recognizer to get the words and recognize them
    with sr.Microphone() as source:
        # Listening to user inputs
        print("Listening")
        # Time for which the recognizer must wait
        r.pause_threshold = 1
        # Taking the source input in variable audio
        audio = r.listen(source)
    try:
        print("Recognizing")
        # Recognizes all the words google can recognize
        query = r.recognize_google(audio, language='en-US')
        print(query)
    except Exception as e:
        print("Unable to recognize your word please try again")
        print(e)
        return None
    return query

def time_now():
# Time displaying function
    time = datetime.datetime.now().strftime("%I:%M%S")# %12 hr format %I and 24 hr format %H
    speak("The Current Time is: ")
    speak(time)

def date_now():
# Date displaying function
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    date = datetime.datetime.now().day
    speak("Current Date is: ")
    speak(date)
    speak(month)
    speak(year)

def wish_me():
    speak("Hello Ameya, Welcome ")
    # time_now()
    # date_now()
    hour = datetime.datetime.now().hour
    if hour >= 6 and hour < 12:
        speak("Good Morning")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon")
    elif hour >= 18 and hour <24:
        speak("Good Evening")
    else:
        speak("Good Night")

    speak("Jarvis at your Service")

def jokes():
    speak(pyjokes.get_jokes())

def screenshot():
    img = pyautogui.screenshot()
    img.save("s1.png")

def sendEmail(to,content):
    # need to enable low security apps to send email
    # from smtp server chnage in gmail settings

    # protocol hostname and connection then login and send email close the server

    # server variable to choose the mailing service
    server = smtplib.SMTP("smtp.gmail.com",587)#587 port number
    # smtp is a way to transfer email from one server to another
    server.ehlo()
    # echlo assigns a hostname to the server to identify itself
    server.starttls()
    # puts the connection to SMTPserver

    server.login('karansingh2020222@gmail.com',"LOR@1234")
    # login to gmail using a low security account
    server.sendmail('karansingh@gmail',to,content)
    server.close()

def take_notes():
    speak("What to write sir?")
    notes = take_command()
    file = open("notes.txt","w")
    speak("Should date and time be included?")
    ans = take_command()
    if "yes" or "sure" or "Yes" or "Sure" in ans:
        strtime = datetime.datetime.now().strftime("%H:%M:%S")
        file.write(strtime)
        file.write(notes)
        speak("Done taking notes")
    else:
        file.write(notes)

def performance():
    #pass
    cpu_val = str(psutil.cpu_percent())
    speak("current cpu at: "+ cpu_val)
    battery_current = psutil.sensors_battery()
    speak("Battery percentage is at: ")
    speak(battery_current)

if __name__ == "__main__":

    clear = lambda: os.system("cls")

    clear()
    wish_me()
    while True:
        # Take user input as comand to function
        query_user = take_command().lower()
        speak(query_user)
        if "time" in query_user:
            time_now()

        elif "date" in query_user:
            date_now()
        # Searching on wikipedia

        elif "wikipedia" in query_user:
            speak("Searching")
            result_1 = wikipedia.summary(query_user,sentences = 1)
            #sentences is number of lines
            print(result_1)
            speak(result_1)

        elif "email" in query_user:
            try:
                speak("What should I say?")
                content = take_command()
                receiver_email = input("Enter receiver email: ")
                sendEmail(receiver_email,content)
                speak(content)
                speak("Email has been sent")

            except Exception as e:
                print(e)
                print("Unable to send an email")


        elif "where is" in query_user:
            location = query_user.replace("where is","")
            wb.open_new_tab("https://www.google.com/maps/place/"+location)


        elif "screenshot" or "capture" in query_user:
            screenshot()

        elif "CPU" or "Cpu" or "perfomance" or "battery" or "Battery" in query_user:
            performance()

        elif "Go offline" or "go offline" in query_user:
            speak("Going offline sir!")
            quit()

        elif "Jokes" or "jokes" in query_user:
            jokes()

        elif "search on google" or "go to google" in query_user:
            speak("What to search on google?")
            search_val = take_command().lower()
            speak("Searching...")
            wb.open("https://www.google.com/search?q="+search_val)

        elif "log out" in query_user:
            os.system("shutdown -1")
        elif "restart" in query_user:
            os.system("shutdown /r/t 1")
        elif "shutdown" in query_user:
            os.system("shutdown /s /t 1")

        elif "Search on net" or "Search on internet" in query_user:
            try:
                speak("What to search")
                explorer_path ='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe %s'
                #explorer_path = 'C:/ProgramData/Microsoft/Windows/Start Menu/Programs/chrome.exe %s'
                search = take_command().lower()
                wb.get(explorer_path).open_new_tab(search+".com")
                #.com ending sites searchced

            except Exception as e:
                print("Unable to search")

        elif "Go to Youtube" or " go to youtube" in query_user:
            speak(" What should I search on youtube")
            search_term = take_command().lower()
            speak("Youtube here we come!")
            explorer_path = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe %s'
            wb.get(explorer_path).open('https://www.youtube.com/results?search_query='+search_term)

        elif "songs" or "music" in query_user:
            music_path = "D:\songs"
            # where songs are stored
            music = os.listdir(music_path)
            # music has the list of all files in the directory
            speak("what to play")
            speak("select a number")
            ans = take_command().lower()
            if 'number' in ans:
                no = int(ans.replace('number',''))
            # no user must give an integer and then start the given file
            if "you choose" or "random" in ans:
                no = random.randint(0,len(music))

            os.startfile(os.path.join(music_path,music[no]))
        elif "notes" or "Notes" in query_user:
            speak("Taking notes sir")
            take_notes()

        elif "show notes" in query_user:
            speak("Showing notes")
            file = open("notes.txt","r")
            print(file.read())
            speak(file.read())
        elif "MS word" or "ms word" or "word" in query_user:
            speak("Opening MS Word")
            ms_word = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Word 2016"
            os.startfile(ms_word)



        elif "news" in query_user:
            speak("Getting news")
            json_obj = urlopen("https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=9a0fdb1691714df1b5bf078282f2d63b")
            speak("Here are the top business headlines")
            print("-----Top Headlines------")
            i = 1
            data = json.load(json_obj)
            # This is ajson object like dictationary
            for item in data["articles"]:
                print(str(i)+"."+item['title']+"\n")
                print(item["description"]+"\n")
                speak(item["title"])
                i = i+1
        elif "stop listening" in query_user:
            c = 0
            while c == 0:
                speak("For how many seconds you want me to stop listening sir?")
                time_wait = int(take_command())
                time.sleep(time_wait)
                speak("can I start listening again sir?")
                ans = take_command()
                if "yes" in ans:
                    c = 1
        elif "what I told you" or "word remembered" or "what you remember" in query_user:
            speak("I remember that sir")
            ans2 = open("memory.txt","r")
            speak("You asked me to remember"+ans2.read())

        elif "remember" or "remember that" in query_user:
            speak("What should I remember")
            ans = take_command()
            speak("You asked me to remember" + ans)
            remember = open("memory.txt","w")
            remember.write(ans)
            remember.close()



    '''
        elif "calculate" in query_user:
            client = wolframalpha.Client(wolframalpha_app_id)
            idx = query_user.lower().split().index("calculate")
            query = query_user.split()[idx + 1:]
            res = client.query("".join(query))
            answer = next(res.results).text
            print("Answer is:"+answer)
            speak("Answer is :"+answer)
        '''