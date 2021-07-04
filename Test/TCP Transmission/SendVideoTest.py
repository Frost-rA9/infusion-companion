import socket
import cv2
import numpy
import time
import sys


def SendVideo():
    host = socket.gethostname()
    address = (host, 8002)
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    capture = cv2.VideoCapture("../../Resource/CAER/TEST/Anger/0001.avi")
    ret, frame = capture.read()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    while ret:
        time.sleep(0.01)
        result, encode_img = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(encode_img)
        stringData = data.tostring()
        client_socket.send(str.encode(str(len(stringData)).ljust(16)))
        client_socket.send(stringData)
        receive = client_socket.recv(1024)
        if len(receive):
            print(str(receive, encoding='utf-8'))
        ret, frame = capture.read()
        if cv2.waitKey(30) == 27:
            break
    client_socket.close()


if __name__ == '__main__':
    SendVideo()
