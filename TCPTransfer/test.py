from threading import Thread
from Server import Server
import time

if __name__ == '__main__':
    server = Server()
    t = Thread(target=server.StaticHandle)
    t.start()
    while True:
        time.sleep(1)
        print(server.get_port_list())
