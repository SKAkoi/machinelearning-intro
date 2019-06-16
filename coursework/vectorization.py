import numpy as np 

a = np.array([1,2,4,5])
print(a)

#import time to track how long operations take
import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)

#Vectorized version
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(c)
print("Vectorized version: " + str(1000*(toc-tic)) + "ms")

#non-vectorized version
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()
print(c)
print("Non-vectorized version: " + str(1000*(toc-tic)) + "ms")

