import socket
import cv2
import numpy
from multiprocessing import Process
from threading import Thread


class Server:
    def __init__(self):
        # 接收数据量大小
        self.BUFFER_SIZE = 1024
        # 静态socket，用于监听后续发送视频数据的端口
        self.static_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 0.0.0.0代表监听局域网下所有IPV4地址的连接
        self.host = '0.0.0.0'
        # 静态端口，用于接收后续端口
        self.static_port = 6400
        # 绑定静态socket
        self.static_socket.bind((self.host, self.static_port))
        # 设置监听数
        self.static_socket.listen(1)
        self.port = None
        self.port_list = []

    def StaticHandle(self):
        while True:
            # 等待连接
            client_socket, client_address = self.static_socket.accept()
            # 连接成功后接收后续用于传输数据的端口
            self.ReceivePort(client_socket)

    def handle(self):
        # 建立新的套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 与后续传输视频的端口绑定
        server_socket.bind((self.host, int(self.port)))
        server_socket.listen(1)
        client_socket, client_address = server_socket.accept()
        self.ReceiveVideo(client_socket, self.port)
        server_socket.close()

    def ReceivePort(self, client_socket):
        # 接收后续传输数据的端口耨
        receive_port = client_socket.recv(self.BUFFER_SIZE)
        self.port = receive_port.decode("utf-8")
        self.port_list.append(self.port)

        # 将接收到的信息发送回客户端进行确认
        return_data = "confirm"
        client_socket.send(return_data.encode('utf-8'))
        handle_thread = Thread(target=self.handle)
        handle_thread.start()
        # self.handle()

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

        while True:
            # 首先接收客户端发送的数据长度，16代表接收长度
            try:
                length = Receive(client_socket, 16)
            except socket.error:
                break
            # 若客户端没有向服务端发送数据长度，停止接收
            if length is None:
                self.port_list.remove(name)
                break
            # 接收字符串格式数据
            stringData = Receive(client_socket, int(length))
            # 将字符串格式数据转换为numpy矩阵格式数据
            data = numpy.frombuffer(stringData, numpy.uint8)
            # 对numpy矩阵格式数据进行解码得到图片
            decode_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            # 使用opencv显示图像，在此处进行修改
            cv2.imshow(f'{name}', decode_img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    server = Server()
    t = Thread(target=server.StaticHandle)
    t.start()