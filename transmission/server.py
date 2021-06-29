import socket
import sys

if __name__ == '__main__':
    socketserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()

    port = 9999

    socketserver.bind((host, port))
    socketserver.listen(5)

    while True:
        socketclient, addr = socketserver.accept()
        print("连接地址: %s" % str(addr))
        msg = '测试通过' + "\r\n"
        socketclient.send(msg.encode('utf-8'))
        socketclient.close()
