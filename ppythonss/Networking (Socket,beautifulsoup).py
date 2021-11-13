import os
os.chdir(r"A:\Data science")
import socket
import bs4
from bs4 import BeautifulSoup
import urllib.request, urllib.parse,urllib.error
import ssl
import re
import json
import xml.etree.ElementTree as ET
#mysock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#mysock.connect(('data.pr4e.org',80))
#cmd='GET  http://data.pr4e.org/intro-short.txt HTTP/1.0\r\n\r\n'.encode()
#mysock.send(cmd)

#while True:
    #data = mysock.recv(512)
    #if (len(data)<1):
        #break
    #print(data.decode()) #print(line.decode().strip())
#mysock.close()


#fhand=urllib.request.urlopen("http://data.pr4e.org/romeo.txt")
#counts=dict()
#for line in fhand:
      #words=line.decode().strip()
      #print(words)
      #for letter in words:
          #counts[letter]=counts.get(letter,0)+1
#print(counts)
#ctx=ssl.create_default_context()
#ctx.check_hostname=False
#ctx.verify_mode=ssl.CERT_NONE
#url="http://www.dr-chuck.com/page1.htm"
#html=urllib.request.urlopen(url, context=ctx).read()
#soup=BeautifulSoup(html,"html.parser")
#Beautiful Soup object is soup
#tags=soup("a")
#for tag in tags:
    #print(tag.get("href",None))

#Week 4 A
#ctx=ssl.create_default_context()
#ctx.check_hostname=False
#ctx.verify_mode=ssl.CERT_NONE
#url="http://python-data.dr-chuck.net/comments_353539.html"
#html=urllib.request.urlopen(url, context=ctx).read()
#soup=BeautifulSoup(html,"html.parser")
#tags=soup("span")
#sum=0
#lst=list()
#for i in tags:
    #print(i)
    #for j in i:
        #a=j.split(">")
        #b=int(a[0])
        #lst.append(b)
#print(lst)
#print(sum(lst))


#Week 4 B
#from bs4 import BeautifulSoup
#import urllib.parse,urllib.request,urllib.error
#import ssl
#ctx=ssl.create_default_context()
#ctx.check_hostname=False
#ctx.verify_mode=ssl.CERT_NONE

#url='http://python-data.dr-chuck.net/known_by_Fikret.html '
#count=5
#pos=3
#for i in range(count) :
 #html=urllib.request.urlopen(url,context=ctx).read()
 #soup=BeautifulSoup(html,"html.parser")
 #tags=soup("a")
 #s=[]
 #t=[]
 #for tag in tags:
    #x=tag.get("href",None)
    #s.append(x)
    #print(x)
    #y=tag.text
    #print(y)
    #t.append(y)
 #print(s[pos-1])
 #print(t[pos-1])
 #url=s[pos-1]

#XML
import xml.etree.ElementTree as ET
data='''
 <person>
    <name>Ameya</name>
    <phone type="intl">
    +91 9930121139
    </phone>
    <email hide="yes"/></person>'''
#tree=ET.fromstring(data)
#print("Name: ",tree.find('name').text)
#print("Attr: ",tree.find('email').get("hide"))
#import urllib.request, urllib.parse,urllib.error
#input_url=urllib.request.urlopen("http://python-data.dr-chuck.net/comments_353536.xml")
#data=input_url.read()
#tree=ET.fromstring(data)
#sum=0
#lst=tree.findall('comments/comment')
#for item in lst:
    #print("Count: ", item.find('count').text)
    #a=int(item.find('count').text)
    #sum=sum+a
    #print(type(a))
#print(sum)

# WEEK 3
'''    
mysock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
mysock.connect(("data.pr4e.org",80))
cmd="GET http://data.pr4e.org/page1.htm HTTP/1.0\r\n\r\n".encode()
mysock.send(cmd)
while True:
    data=mysock.recv(512)
    if(len(data)<1):
        break
    print(data.decode())
mysock.close()
'''
'''
#WEEK 4
ctx=ssl.create_default_context()
ctx.check_hostname=False
ctx.verify_mode=ssl.CERT_NONE
fhandle=urllib.request.urlopen("http://data.pr4e.org/page1.htm").read()
soup=BeautifulSoup(fhandle,"html.parser")
tags=soup("a")
for tag in tags:
    print(tag.get("href",None))
'''
'''
Week 4 A
fhandle=urllib.request.urlopen("http://python-data.dr-chuck.net/comments_42.html").read()
soup=BeautifulSoup(fhandle,"html.parser")
tags=soup("span")
a=[]
for tag in tags:
    #print(tag)
    for j in  tag:
     j.split(">")
     a.append(int(j))
print(sum(a))
'''
#week 4 B
"""
pos=3
url = "http://python-data.dr-chuck.net/known_by_Fikret.html"
for i in range(4):
 fhandle=urllib.request.urlopen(url).read()
 soup=BeautifulSoup(fhandle,"html.parser")

 tags=soup("a")
 s=[]
 t=[]
 for tag in tags:
    x=tag.get("href", None)
    #print(x)
    s.append(x)
    y=tag.text
    #print(y)
    t.append(y)

 #print(s[pos - 1])
 url = s[pos - 1]
 print(url)
"""

#Week 5
'''
url="http://python-data.dr-chuck.net/comments_42.xml"
fhandle=urllib.request.urlopen(url).read()
soup=BeautifulSoup(fhandle,"html.parser")

#print(soup)
tree=et.fromstring(fhandle)
print(fhandle)
a=tree.findall("comments/comment")
c=[]
for i in a:
    b=i.find("count").text
    print(b)
    c.append(int(b))
print(sum(c))
'''


#week6
'''
data='''
{"name":"Ameya",
"phone":"+91 9930121139",
"address":{"Country":"India", "City":"Mumbai"},
"email":{"hide":"yes"}}
'''
info=json.loads(data)
print("Name: ", info["name"])
print("Number: ",info["phone"])
print("Address: ",info["address"]["Country"])
print("Hide: ",info["email"]["hide"])
'''

'''
#week 6 A
url=" http://python-data.dr-chuck.net/comments_353540.json"
fhandle=urllib.request.urlopen(url).read()
#fhandle1=list(fhandle)
#print(type(fhandle1))
info=json.loads(fhandle)
b=list()
print(type(info))
for i in info["comments"]:
 print(i)
 print(i["count"])
 b.append(i["count"])
print(sum(b))
'''
#Week 6 B
'''
url1="http://maps.googleapis.com/maps/api/geocode/json?"
while True:
    address = input('Enter location: ')
    if len(address) < 1 : break

    url = url1 + urllib.parse.urlencode({'sensor':'false', 'address': address})
    print ('Retrieving', url)
    uh = urllib.request.urlopen(url)
    data = uh.read()
    print ('Retrieved',len(data),'characters')

    try:
        js = json.loads(str(data))
    except:
        js = None
    if not js or "Status" not in js or js['status'] != 'OK':
        print('==== Failure To Retrieve ====')
        print(data)
        continue
    print(json.dump(js,indent=4))
    a=js["results"][0]["place_id"]
    print(a)
# Ann Arbor,MI  BITS Pilani  IIT KANPUR
'''
'''
working solution    Week 6
serviceurl = "http://python-data.dr-chuck.net/geojson?"
# This API only accepts the university in a list of its accepted ones.
# This API uses the same parameters (sensor and address) as the Google API.
# This API also has no rate limit so you can test as often as you like.
#uk If you visit the URL with no parameters, you get a list of all of the address values which can be used with this API.

address_input = input("Enter location: ")
params = {"sensor": "false", "address": address_input}
url = serviceurl + urllib.parse.urlencode({'sensor':'false', 'address':  address_input})
print("Retrieving ", url)
data = urllib.request.urlopen(url).read().decode('utf-8')
print('Retrieved', len(data), 'characters')
js = json.loads(data)
print(json.dumps(js,indent=4))
place_id = js["results"][0]["place_id"]
print("Place id", place_id)

#   USC   Ann Arbor,MI  BITS Pilani  IIT KANPUR  South Federal University  The University of South Africa

#USC Ann Arbor,MI  BITS Pilani  IIT KANPUR  South Federal University  The University of South Africa
'''

'''
class calc:
    x=0
    def add(self):
      self.x  = self.x + 1
      print(self.x)

obj=calc()
obj.add()
'''

'''
Week 2 Database systems
import sqlite3

conn = sqlite3.connect('emaildb.sqlite')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS Counts')

cur.execute('''
CREATE TABLE Counts (email TEXT, count INTEGER)''')

fname = input('Enter file name: ')
if (len(fname) < 1): fname = 'mbox-short.txt'
fh = open(fname)
for line in fh:
    if not line.startswith('From: '): continue
    pieces = line.split()
    email = pieces[1]
    cur.execute('SELECT count FROM Counts WHERE email = ? ', (email,))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (email, count)
                VALUES (?, 1)''', (email,))
    else:
        cur.execute('UPDATE Counts SET count = count + 1 WHERE email = ?',
                    (email,))
conn.commit()

# https://www.sqlite.org/lang_select.html
sqlstr = 'SELECT email, count FROM Counts ORDER BY count DESC LIMIT 10'

for row in cur.execute(sqlstr):
    print(str(row[0]), row[1])

cur.close()
#mbox-short.txt
'''
'''
import sqlite3
import urllib

#Connecting to the file in which we want to store our db
conn = sqlite3.connect('emaildb34.sqlite')
cur = conn.cursor()

#Deleting any possible table that may affect this assignment
cur.execute('''
DROP TABLE IF EXISTS Counts''')

#Creating the table we're going to use
cur.execute('''
CREATE TABLE Counts (org TEXT, count INTEGER)''')

#Indicating the file (URL in this case) from where we'll read the data

fh = open("mbox-short.txt")

#Reading each line of the file
for line in fh:

    #Finding an email address and splitting it into name and organization
    if not line.startswith('From: ') : continue
    pieces = line.split()
    email = pieces[1]
    (emailname, organization) = email.split("@")
    print (email)

    #Updating the table with the correspondent information
    cur.execute('SELECT count FROM Counts WHERE org = ? ', (organization, ))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (org, count)
                VALUES ( ?, 1 )''', ( organization, ) )
    else :
        cur.execute('UPDATE Counts SET count=count+1 WHERE org = ?',
            (organization, ))

# We commit the changes after they've finished because this speeds up the
# execution and, since our operations are not critical, a loss wouldn't suppose
# any problem
conn.commit()

# Getting the top 10 results and showing them
sqlstr = 'SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10'

print ("Counts:")
for row in cur.execute(sqlstr) :
    print (str(row[0]), row[1])

#Closing the DB
cur.close()

'''
'''
import sqlite3
import xml.etree.ElementTree as Et
conn=sqlite3.connect("M1.sqlite")
cur=conn.cursor()
cur.executescript('''
Drop table if exists Artist;
Drop table if exists Album;
Drop table if exists Genre;
Drop table if exists Track;
''')
cur.executescript(
    '''
    Create table Artist(
    id Integer Not null primary key Autoincrement Unique,
    name text unique
    );
    Create table Album(
      id Integer Not null primary key Autoincrement Unique,
      title text unique,
      artist_id integer
       );
   Create table Genre(
     id Integer Not null primary key Autoincrement Unique,
    name text unique
   );
   Create table Track(
   id Integer Not null primary key Autoincrement Unique,
   title text unique,
   rating integer,
   len integer,
   count integer,
   album_id integer,
   genre_id integer
   );
    ''')
fhandle1=open("library.xml")
fhandle=fhandle1.read()
data1=Et.fromstring(fhandle)
data=data1.findall('dict/dict/dict')
print('Dict count:', len(data))

def lookup(d, key):
    found = False
    for child in d:
        if found :
            return child.text
        if child.tag == 'key' and child.text == key :
            found = True
    return None
for i in data:
    if (lookup(i,"Track ID")is None):continue
    name = lookup(i,"Name")
    artist=lookup(i, "Artist")
    album=lookup(i,"Album")
    genre=lookup(i,"Genre")
    rating=lookup(i,"Rating")
    count=lookup(i,"Play Count")
    len=lookup(i,"Total Time")

    if name is None or artist is None or album is None:
        continue
    print('Genre Name :\t', genre)
    cur.execute('''INSERT or IGNORE  into Artist(name) values(?)''',(artist,))
    cur.execute('select id from Artist where name=?',(artist,))
    artist_id=cur.fetchone()[0]
    cur.execute('''INSERT or IGNORE  into Album(title,artist_id) values(?,?)''',(album,artist_id))
    cur.execute('Select id from Album where title=?',(album,))
    album_id=cur.fetchone()[0]
    cur.execute('''INSERT or IGNORE into Genre(name) values(?)''',(genre,))
    cur.execute('select id from Genre where name=?',(genre,))
    try:
        genre_id = cur.fetchone()[0]
    except:
        genre_id = ''
    cur.execute('''INSERT or REPLACE INTO Track(title,rating,len,count,album_id,genre_id) values(?,?,?,?,?,?)''',(name,rating,len,count,album_id,genre_id))

conn.commit()
s
'''
"""
import sqlite3
import json
conn=sqlite3.connect("Rosterdb")
cur=conn.cursor()
cur.executescript("""
Drop table if exists user;
Drop table if exists member;
Drop table if exists course;
Create  table user(
id integer not null primary key autoincrement unique,
email Text,
name Text
);

create table course(
id integer not null primary key autoincrement unique,
title text
);
create table member(
user_id integer,
role integer,
course_id
);
""")
fhandle =urllib.request.urlopen("https://raw.githubusercontent.com/AlexGascon/Using-Databases-with-Python---Coursera/master/Unit%204%20-%20Many%20to%20many%20relationships%20in%20SQL/roster_data.json").read()
#fhandle=open("abc.json")
print(type(fhandle))
data=json.loads(fhandle)
print(type(data))

for entry  in data :
    user=entry[0]
    course=entry[1]
    instructor=entry[2]
    cur.execute("""Insert or ignore into user(name) values(?)""",(user,))
    cur.execute("Select id from user where name=?",(user,))
    user_id=cur.fetchone()[0]#selects first entry eliminates duplicates

    cur.execute("""Insert or ignore into course(title) values(?)""", (course,))
    cur.execute("Select id from course where title=?", (course,))
    course_id = cur.fetchone()[0]

    cur.execute('''Insert or replace into member(user_id,course_id,role) values(?,?,?)''',(user_id,course_id,instructor))
'''