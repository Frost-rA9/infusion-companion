# import datetime
# import time
#
# start = time.time()
# time.sleep(1)
# end = time.time()
# spend = end - start
# print(type(spend), spend)
#
# a = True
# print(not a)
# a = not a
# print(a)
# a = "hhh"
# b = str(a)
# print(b)

import numpy as np
a = np.zeros((10), np.int)
a[0] = 1
a[0] = sum(a == 0)
a = a * 10
print(a)
print(sum(a == 0), type(sum(a == 0)))