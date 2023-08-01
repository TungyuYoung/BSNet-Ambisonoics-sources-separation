import numpy as np

class Coordinates:  # 坐标

    def __init__(self, input, type='cartesian'):

        input = np.asarray(input)

        if type == 'cartesian':  # 笛卡尔坐标系
            self.cart = input

        if type == 'sphericalDeg':  # 球面坐标系
            self.cart[0] = np.cos(input[0]) * np.sin(input[1])
            self.cart[1] = np.sin(input[0]) * np.sin(input[1])
            self.cart[2] = np.cos(input[1])

        self.x = self.cart[0]
        # print(self.x)
        self.y = self.cart[1]
        # print(self.y)
        self.z = self.cart[2]
        # print(self.z)

        self.cartNorm = self.cart / np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)  # 归一化坐标
        # print(self.cartNorm)

        self.azi = np.arctan2(self.y, self.x)  # 方位角 Φ
        self.r = np.sqrt(np.sum(self.cart ** 2))  # 半径
        self.zen = np.arccos(self.z / self.r)  # 俯仰角 θ
        self.ele = np.pi / 2 - self.zen  # pi / 2 - 俯仰角

        self.aziEle = np.hstack((self.azi, self.ele))  # 拼接
        self.aziZen = np.hstack((self.azi, self.zen))  # 拼接

    def __str__(self):
        return "azi=" + str(self.azi * 180 / np.pi) + " ele=" + str(self.ele * 180 / np.pi) + " r= " + str(self.r)

    def greatCircleDistanceTo(self, c2):

        if (self.cartNorm == c2.cartNorm).all():
            return 0

        else:
            phi = np.arccos(np.inner(self.cartNorm, c2.cartNorm))
            return phi
