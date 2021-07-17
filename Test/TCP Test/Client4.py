import socket
import cv2
import numpy
import time


class Client:
    def __init__(self):
        # 接收数据量大小
        self.BUFFER_SIZE = 1024
        # 静态socket，用于监听后续的端口
        self.static_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # # 获取主机名
        # self.hostname = socket.gethostname()
        # # 通过主机名获取到IP地址
        self.host = socket.gethostbyname('FROST-DESKTOP')
        # host位服务端的IP地址
        # self.host = '192.168.1.242'
        # 静态端口
        self.static_port = 6400
        # 客户端socket，用于后续发送视频数据
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 端口
        self.port = 6704

    def SendPort(self):
        try:
            self.static_socket.connect((self.host, self.static_port))
        except socket.error as msg:
            print(msg)
        self.static_socket.send(str(self.port).encode('utf-8'))
        receive_message = self.static_socket.recv(self.BUFFER_SIZE)
        confirm_message = receive_message.decode("utf-8")
        if confirm_message == "confirm":
            self.SendVideo()

    def SendVideo(self):
        try:
            self.client_socket.connect((self.host, self.port))
        except socket.error as msg:
            print(msg)
        capture = cv2.VideoCapture("../../Resource/TestVideo/0004.avi")
        ret, frame = capture.read()
        # 压缩参数，0-100，100为最好，目前没有测试未压缩传输
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        while ret:
            time.sleep(0.015)
            # 对图像进行压缩
            result, encode_img = cv2.imencode('.jpg', frame, encode_param)
            # 将压缩后的图像转换为numpy矩阵数据
            data = numpy.array(encode_img)
            # 将numpy矩阵数据转换为字节数据
            bytesData = data.tobytes()
            # 向服务端发送数据长度
            self.client_socket.send(str.encode(str(len(bytesData)).ljust(16)))
            # 向服务端发送数据
            self.client_socket.send(bytesData)
            ret, frame = capture.read()
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        self.client_socket.close()


if __name__ == '__main__':
    client = Client()
    client.SendPort()
