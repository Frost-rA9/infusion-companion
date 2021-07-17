import datetime
import time

start = time.time()
time.sleep(1)
end = time.time()
spend = end - start
print(type(spend), spend)