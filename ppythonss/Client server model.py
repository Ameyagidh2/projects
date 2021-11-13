# CLIENT
'''
import socket
mysock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
mysock.connect(('www.w3.org',80))
cmd='GET /https://www.w3.org/1999/xhtml/HTTP/1.1\r\nhost:103.42.192.125\r\n\r\n'.encode()
mysock.send(cmd)

while True:
    data = mysock.recv(512)
    if (len(data)<1):
        break
    print(data.decode()) #print(line.decode().strip())
mysock.close()
'''
'''
import sys
import socket

try:
    mysock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
except socket.error:
    print("Failed to create socket")
    sys.exit()

try:
    host=socket.gethostbyname("www.w3.org")
except socket.gaierror:
    print("Failed to get host names")
    sys.exit()
msg="GET /https://www.w3schools.com/sql/1.1\r\n\r\n\r\n".encode()

mysock.connect((host,80))
try:
 mysock.sendall(msg)
except  socket.error:
    print("Failed to create socket")
    sys.exit()
data=mysock.recv(1000)
print(data.decode())
mysock.close()
'''
import socket

HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = "192.168.56.1"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

send("Hello World!")
input()
send("Hello Everyone!")
input()
send("Hello Tim!")

send(DISCONNECT_MESSAGE)