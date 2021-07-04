import socket
import time
import cv2
import numpy


def ReceiveVideo():
    host = socket.gethostname()
    address = (host, 8002)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(address)
    server_socket.listen(1)
    def Receive(sock, count):
        buf = b''
        while count:
            new_buf = sock.recv(count)
            if not new_buf:
                return None
            buf += new_buf
            count -= len(new_buf)
        return buf

    connect, address = server_socket.accept()
    print('connect from:' + str(address))
    while 1:
        start = time.time()
        length = Receive(connect, 16)
        stringData = Receive(connect, int(length))
        data = numpy.frombuffer(stringData, numpy.uint8)
        decode_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2.imshow('SERVER', decode_img)
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        connect.send(bytes(str(int(fps)), encoding='utf-8'))
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    server_socket.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ReceiveVideo()
