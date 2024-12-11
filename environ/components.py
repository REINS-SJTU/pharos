from typing import Tuple

import numpy as np

from environ.utils import Vector3


def spinner(alpha: float, beta: float, gamma: float) -> np.ndarray:
    x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return z @ y @ x


class Zone:
    def __init__(self, x: Tuple[float, float], y: Tuple[float, float], z: Tuple[float, float]) -> None:
        assert np.greater_equal(x, 0).all()
        assert np.greater_equal(y, 0).all()
        assert np.greater_equal(z, 0).all()
        self.__x = x
        self.__y = y
        self.__z = z

    def efficiency(self, c: float):
        return sum(self.__x) * sum(self.__y) * sum(self.__z) / (c * 2) ** 3

    def constraint(self, direction: Vector3) -> float:
        with np.errstate(divide='ignore'):
            constraints = (
                -self.__x[0] / direction.x,
                self.__x[1] / direction.x,
                -self.__y[0] / direction.y,
                self.__y[1] / direction.y,
                -self.__z[0] / direction.z,
                self.__z[1] / direction.z,
            )
        factor = min(i for i in constraints if i >= 0)
        return factor


class Vehicle:
    def __init__(self, v: float, a: float, priority: float, tick: float) -> None:
        assert 0 < v
        assert 0 < a
        assert 0 < priority
        assert 0 < tick
        self.__v = v
        self.__a = a
        self.__priority = priority
        self.__tick = tick
        self.__odometer = 0.0
        self.direction = Vector3(0, 0, 0)
        self.position = Vector3(0, 0, 0)
        self.speed = 0
        self.offline = False

    @property
    def v(self) -> float:
        return self.__v

    @property
    def a(self) -> float:
        return self.__a

    @property
    def tick(self) -> float:
        return self.__tick

    @property
    def priority(self) -> float:
        return self.__priority

    @property
    def odometer(self) -> float:
        return self.__odometer

    @property
    def boundary(self) -> float:
        return self.v * self.tick

    def move(self, zone: Zone, randomness=0.05) -> float:
        alpha, beta, gamma = np.random.normal(0.0, self.tick * randomness, size=3)
        rotated = self.direction.t @ spinner(alpha, beta, gamma)
        self.direction = Vector3(*rotated)

        constraint = zone.constraint(self.direction)
        vi = self.speed
        a = 2 / self.tick ** 2 * (constraint - vi * self.tick)
        a = np.clip(a, -min(vi / self.tick, self.a), self.a)

        if a > 0 and (t := abs(self.v - vi) / a) < self.tick:
            d1 = vi * t + 1 / 2 * a * t ** 2
            d2 = self.v * (self.tick - t)
            moved, vf = d1 + d2, self.v
        else:
            d = vi * self.tick + 1 / 2 * a * self.tick ** 2
            vf = vi + a * self.tick
            moved, vf = d, np.clip(vf, 0, None)

        self.__odometer += moved
        self.position += self.direction * moved
        self.speed = vf

        return moved


class Human:
    def __init__(self, v: float, theta: float, phi: float, tick: float) -> None:
        assert 0 <= v
        assert -np.pi <= theta <= np.pi
        assert 0 <= phi <= np.pi
        self.__v = v
        self.__theta = theta
        self.__phi = phi
        self.__tick = tick
        self.position = Vector3(0, 0, 0)
        self.direction = Vector3(0, 0, 0)
        self.offline = False

    @property
    def v(self) -> float:
        return self.__v

    @property
    def tick(self) -> float:
        return self.__tick

    def move(self, randomness=0.5) -> None:
        alpha, beta, gamma = np.random.normal(0.0, self.tick * randomness, size=3)
        rotated = self.direction.t @ spinner(alpha, beta, gamma)
        self.direction = Vector3(*rotated)
        self.position += self.direction * self.tick * self.v

    def observe(self, vehicle: Vehicle) -> float:
        relation = vehicle.position - self.position
        h = abs(self.__theta - relation.theta)
        v = abs(self.__phi - relation.phi)
        c = np.pi / 180
        if h > np.pi:
            h = 2 * np.pi - h

        visible = v < 60 * c and h < 80 * c
        distance = relation.magnitude
        level = abs(vehicle.direction.z) <= 0.2

        return 0.0 \
            + 0.25 * (distance < 8 and not level) \
            + 0.25 * (distance < 8 and not visible) \
            + 0.25 * (distance < 3 and vehicle.speed > 0.5) \
            + 0.25 * (distance < 3)
