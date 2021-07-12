import cv2
import numpy
import socket
from multiprocessing import Process
import threading


class Server:

    def __init__(self):
        # 创建socket套接字
        self.server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置IP地址以及端口 0.0.0.0代表监听局域网下所有的IPV4地址
        self.host = "0.0.0.0"
        self.port_1 = 1721
        self.port_2 = 1722
        # 绑定socket套接字
        self.server_socket_1.bind((self.host, self.port_1))
        self.server_socket_2.bind((self.host, self.port_2))
        # 设置监听客户端个数，每个端口仅仅监听一个客户端连接
        self.server_socket_1.listen(1)
        self.server_socket_2.listen(1)

    def ReceiveVideo(self, client_socket, name):
        # 接收字节流
        def Receive(sock, count):
            buf = b''
            while count:
                new_buf = sock.recv(count)
                if not new_buf:
                    return None
                buf += new_buf
                count -= len(new_buf)
            return buf

        while True:
            # 首先接收客户端发送的数据长度，16代表接收长度
            length = Receive(client_socket, 16)
            # 若客户端没有向服务端发送数据长度，停止接收
            if length is None:
                break
            # 接收字符串格式数据
            stringData = Receive(client_socket, int(length))
            # 将字符串格式数据转换为numpy矩阵格式数据
            data = numpy.frombuffer(stringData, numpy.uint8)
            # 对numpy矩阵格式数据进行解码得到图片
            decode_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            # 使用opencv显示图像，在此处进行修改
            cv2.imshow(f'{name}', decode_img)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()

    # 监听1号端口
    def handle_1(self):
        while True:
            # 接收1号端口客户端的连接
            client_socket_1, client_info_1 = self.server_socket_1.accept()
            # 接收到连接后开始接收视频图像
            self.ReceiveVideo(client_socket_1, 'camera1')
        self.server_socket_1.close()

    # 监听2号端口
    def handle_2(self):
        while True:
            # 接收2号端口客户端的连接
            client_socket_2, client_info_2 = self.server_socket_2.accept()
            # 接收到连接后开始接收视频图像
            self.ReceiveVideo(client_socket_2, 'camera2')
        self.server_socket_2.close()


if __name__ == '__main__':
    server = Server()
    """
    多进程或多线程监听多个端口
    即同时监听多个端口，达到处理客户端随机连接的效果
    """
    Process1 = Process(target=server.handle_1)
    Process1.start()
    Process2 = Process(target=server.handle_2)
    Process2.start()
    # t1 = threading.Thread(target=server.handle_1)
    # t1.start()
    # t2 = threading.Thread(target=server.handle_2)
    # t2.start()
