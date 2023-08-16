from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z = np.reshape(self.state, (len(self.state), 1))
        x_pred = np.dot(self.A, z) + np.dot(self.B, np.atleast_2d(u))
        y = q - np.dot(self.W, z)
        x_est = np.dot(self.L, np.reshape(y, (len(y), 1)))
        self.state = self.state + np.reshape((x_pred + x_est), (1, len(x_pred))).flatten() * self.Tp

    def get_state(self):
        return self.state
