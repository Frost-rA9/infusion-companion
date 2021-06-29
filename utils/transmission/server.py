import socket
import cv2
import numpy as np


def ReceiveVideo():
    # 客户端地址
    host = '0.0.0.0'
    port = 9999
    # 建立socket 对象
    socketserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定地址
    socketserver.bind((host, port))
    socketserver.listen(5)

    def ReceiveCall(sock, count):
        buf = 'b'
        # 接收TCP数据
        while count:
            new_buf = sock.recv(count)
            if not new_buf:
                return None
            buf += new_buf
            count -= len(new_buf)
        return buf

    connect, address = socketserver.accept()
    print('connect from:' + str(address))
    while True:
        # 获取图片文件长度
        length = ReceiveCall(connect, 16)
        # 获取字符串格式图片数据
        string_data = ReceiveCall(connect, int(length))
        # 将获取到的数据利用numpy格式化
        data = np.frombuffer(string_data, np.uint8)
        # 解码操作
        decode_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # 显示图像
        cv2.imshow('SERVER', decode_img)

        socketserver.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ReceiveVideo()
