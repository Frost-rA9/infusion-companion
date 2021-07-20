import socket

if __name__ == '__main__':
    hostname = socket.gethostname()
    host = socket.gethostbyname(hostname)
    print(host)