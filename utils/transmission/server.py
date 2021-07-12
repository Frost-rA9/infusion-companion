import cv2
import numpy
import socket
import time
from multiprocessing import Process
import threading


class Server:

    def __init__(self):
        self.server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.host = "0.0.0.0"
        self.port_1 = 1721
        self.port_2 = 1722

        self.server_socket_1.bind((self.host, self.port_1))
        self.server_socket_2.bind((self.host, self.port_2))

        self.server_socket_1.listen(1)
        self.server_socket_2.listen(1)

    def ReceiveVideo(self, client_socket, name):
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
            yield decode_img
        #     cv2.imshow(f'{name}', decode_img)
        #     k = cv2.waitKey(10) & 0xff
        #     if k == 27:
        #         break
        #     end = time.time()
        #     seconds = end - start
        #     fps = 1 / seconds
        #     client_socket.send(bytes(str(int(fps)), encoding='utf-8'))
        # cv2.destroyAllWindows()

    def handle_1(self):
        while True:
            client_socket_1, client_info_1 = self.server_socket_1.accept()
            self.ReceiveVideo(client_socket_1, 'camera1')
        self.server_socket_1.close()

    def handle_2(self):
        while True:
            client_socket_2, client_info_2 = self.server_socket_2.accept()
            self.ReceiveVideo(client_socket_2, 'camera2')
        self.server_socket_2.close()


if __name__ == '__main__':
    Server = Server()
    t1 = threading.Thread(target=Server.handle_1)
    t1.start()
    t2 = threading.Thread(target=Server.handle_2)
    t2.start()
