import socket
import cv2
import numpy
import time
import sys


class Client:
    def __init__(self):
        # 获取本地地址
        self.hostname = socket.gethostname()
        self.host = socket.gethostbyname(self.hostname)
        # 设置端口，端口需要和服务端的一个监听项保持一只
        self.port = 1721
        # 创建socket套接字
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def SendVideo(self):
        try:
            # 使用对应的IP地址和端口连接服务端
            self.client_socket.connect((self.host, self.port))
        except socket.error as msg:
            print(msg)
            sys.exit(1)
        # 建立连接成功后开始传输视频数据
        # 函数中的参数可以更改为视频文件地址，填0表示客户端设备的1号摄像头
        capture = cv2.VideoCapture(0)
        # 截取一帧图像，ret代表有无帧，frame为截取到的图像
        ret, frame = capture.read()
        # 压缩参数，0-100，100为最好，目前没有测试未压缩传输
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        while ret:
            time.sleep(0.15)
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
            ret,frame = capture.read()
            if cv2.waitKey(10) == 27:
                break
            self.client_socket.close()


if __name__ == '__main__':
    client = Client()
    client.SendVideo()
