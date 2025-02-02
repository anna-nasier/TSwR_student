import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp, 1.5)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x

        K_d = [[20, 0], [0, 30]]
        K_p = [[20, 0], [0, 25]]

        v = q_r_ddot + K_d @ (q_r_dot - [q1_dot, q2_dot]) + K_p @ (q_r - [q1, q2])

        Tau = self.model.M(x) @ v + self.model.C(x) @ q_r_dot
        return Tau
        # return NotImplementedError()
