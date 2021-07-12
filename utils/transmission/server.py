import cv2
import numpy
import socket
import time
from multiprocessing import Process


class Server:

    def __init__(self):
        self.server_socket_1 = None
        self.server_socket_2 = None

        self.host = "0.0.0.0"
        self.port_1 = 1721
        self.port_2 = 1722



