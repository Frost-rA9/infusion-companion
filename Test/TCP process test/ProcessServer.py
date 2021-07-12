import socket
from multiprocessing import Process
import numpy
import cv2
import time
import threading


# 必须使用多进程而不是多线程！
def ReceiveVideo(client_socket, name):
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
        cv2.imshow(f'{name}', decode_img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        client_socket.send(bytes(str(int(fps)), encoding='utf-8'))
    cv2.destroyAllWindows()


def handle_1(host1, port1):
    server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_1.bind((host1, port1))
    server_socket_1.listen(1)
    while True:
        client_socket_1, client_info_1 = server_socket_1.accept()
        ReceiveVideo(client_socket_1, 'camera1')
    server_socket_1.close()


def handle_2(host2, port2):
    server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_2.bind((host2, port2))
    server_socket_2.listen(1)
    while True:
        client_socket_2, client_info_2 = server_socket_2.accept()
        ReceiveVideo(client_socket_2, 'camera2')
    server_socket_2.close()


if __name__ == '__main__':
    listen_host = "0.0.0.0"
    port_1 = 6001
    port_2 = 6002
    # Process_1 = Process(target=handle_1, args=(listen_host, port_1,))
    # Process_1.start()
    # Process_2 = Process(target=handle_2, args=(listen_host, port_2,))
    # Process_2.start()
    t1 = threading.Thread(target=handle_1, args=(listen_host, port_1))
    t1.start()
    t2 = threading.Thread(target=handle_2, args=(listen_host, port_2))
    t2.start()
