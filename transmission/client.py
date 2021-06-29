import socket
import cv2
import numpy as np
import sys
import time


def SendVideo():
    # 服务端地址
    host = '0.0.0.0'
    port = 9999
    try:
        # 建立socket 对象
        socketclient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 建立连接
        socketclient.connect((host, port))
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    # 获取当前图像
    capture = cv2.VideoCapture(0)
    # 读取一帧图像
    ret, frame = capture.read()
    # 编码参数设置
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    while ret:
        time.sleep(0.15)
        # 图像编码
        result, encode_img = cv2.imencode('.jpg', frame, encode_param)
        # 建立矩阵
        data = np.array(encode_img)
        # 将numpy矩阵转换为字符串数据
        string_data = data.tostring()

        # 传输数据长度
        socketclient.send(str.encode(str(len(string_data)).ljust(16)))
        # 传输数据
        socketclient.send(string_data)
        # 获取服务器的返回值
        receive = socketclient.recv(1024)

        if len(receive):
            print(str(receive, encoding='utf-8'))

        # 获取下一帧图片
        ret, frame = capture.read()

    socketclient.close()


if __name__ == '__main__':
    SendVideo()
