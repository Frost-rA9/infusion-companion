import socket
hostname = socket.gethostname()
host = socket.gethostbyname(hostname)
print(host)