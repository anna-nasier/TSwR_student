import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManiuplatorModel(Tp, 0.1, 0.05), ManiuplatorModel(Tp, 0.01, 0.01), ManiuplatorModel(Tp, 1.0, 0.3)]
        self.i = 0
        self.u = np.zeros((2, 1))

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q1, q2, q1_dot, q2_dot = x
        errors = []
        for model in self.models:
            y = np.dot(model.M(x), self.u) + np.dot(model.C(x), [[q1_dot],[q2_dot]])
            curr_error = np.sum(np.abs([[q1],[q2]]-y))
            errors.append(curr_error)
        idx = errors.index(min(errors))
        self.i = idx
        pass

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        K_d = [[30, 0], [0, 30]]
        K_p = [[30, 0], [0, 30]]
        v = q_r_ddot + K_d @ (q_r_dot - q_dot) + K_p @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
