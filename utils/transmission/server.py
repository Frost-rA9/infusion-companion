import cv2
import numpy
import socket
import time
from multiprocessing import Process


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

