import threading, queue
import numpy as np

if __name__ == '__main__':
    q = np.zeros((1, 2))
    p = np.ones((1, 2))
    print(np.matmul(q.reshape(-1,1), p).shape)
