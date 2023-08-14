import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3 =0.0, r3 = 0.01):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 =r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.d1 = self.l1/2
        self.d2 = self.l2/2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        a = self.m1 * pow(self.d1, 2) + self.I_1 + self.m2 * (pow(self.l1,2) + pow(self.d2,2)) + self.I_2 + self.m3 * (pow(self.l1, 2) + pow(self.l2, 2)) + self.I_3
        b = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        g = self.m2 * pow(self.d2,2) + self.m3 * pow(self.l2, 2) + self.I_2 + self.I_3

        M = np.array([[a + 2 * b * np.cos(q2), g + b * np.cos(q2)], [g + b * np.cos(q2), g]])
        #print(M)
        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        a = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        C =  np.array([[-a * np.sin(q2) * q2_dot, -a * np.sin(q2) * (q1_dot + q2_dot)],[a * np.sin(q2) * q1_dot, 0.0]])
        #print(C)
        return C
        # return NotImplementedError()
