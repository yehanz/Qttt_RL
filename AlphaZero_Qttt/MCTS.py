import numpy as np


class Node:
    def __init__(self, qttt, parent):
        self.parent = parent
        self.qttt = qttt
        self.prob = np.zeros(9*8*2)

        self.N = 0
        self.S = 0
        self.W = 0
        self.Q = 0



class MCTS:
    def __init__(self):
