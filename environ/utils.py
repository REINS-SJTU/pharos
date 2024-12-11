from typing import Tuple

import numpy as np


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return f'({self.x:.2f}, {self.y:.2f}, {self.z:.2f})'

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> 'Vector3':
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)

    @property
    def r(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @property
    def theta(self) -> float:
        return np.arctan2(self.y, self.x)

    @property
    def phi(self) -> float:
        return np.arccos(self.z / self.r)

    @property
    def magnitude(self) -> float:
        return self.r

    @property
    def normalized(self) -> 'Vector3':
        m = self.magnitude
        return Vector3(self.x / m, self.y / m, self.z / m)

    @property
    def t(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


class Spot:
    @staticmethod
    def uniform(low: float, high: float) -> Vector3:
        r = np.random.uniform(low, high)
        theta = np.random.uniform(-np.pi, np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return Vector3(x, y, z)

    @staticmethod
    def normal(loc: float, scale: float) -> Vector3:
        r = np.random.normal(loc, scale)
        theta = np.random.uniform(-np.pi, np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return Vector3(x, y, z)
