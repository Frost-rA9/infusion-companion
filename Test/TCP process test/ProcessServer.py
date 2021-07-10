import socket
from multiprocessing import Process
import numpy
import cv2
import time


# 必须使用多进程而不是多线程！
def ReceiveVideo(client_socket, client_info):
    def Receive(sock, count):
        buf = b''
        while count:
            new_buf = sock.recv(count)
            if not new_buf:
                return None
            buf += new_buf
            count -= len(new_buf)
        return buf

    while 1:
        start = time.time()
        length = Receive(client_socket, 16)
        if length is None:
            break
        stringData = Receive(client_socket, int(length))
        data = numpy.frombuffer(stringData, numpy.uint8)
        decode_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2.imshow('SERVER', decode_img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        client_socket.send(bytes(str(int(fps)), encoding='utf-8'))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address1 = ('0.0.0.0', 6001)
    address2 = ('0.0.0.0', 6002)
    server_socket1.bind(address1)
    server_socket2.bind(address2)
    server_socket1.listen(1)
    server_socket2.listen(1)
    while True:
        client_socket1, client_info1 = server_socket1.accept()
        client_socket2, client_info2 = server_socket2.accept()
        process1 = Process(target=ReceiveVideo, args=(client_socket1, client_info1))
        process1.start()
        process2 = Process(target=ReceiveVideo, args=(client_socket2, client_info2))
        process2.start()
    server_socket1.close()
    server_socket2.close()
