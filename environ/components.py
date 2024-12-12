from typing import Tuple

import numpy as np

from environ.utils import Vector3


def spinner(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    build a spinner that can rotate a vector in three dimensions
    :param alpha: radians for x-axis
    :param beta: radians for y-axis
    :param gamma: radians for z-axis
    :return: a numpy array
    """
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
        """
        the rectangular exclusive zone that vehicles are restricted in
        the zone always align with the cartesian axis
        :param x: magnitude of restrictions in negative and positive x-axis
        :param y: magnitude of restrictions in negative and positive y-axis
        :param z: magnitude of restrictions in negative and positive z-axis
        """
        assert np.greater_equal(x, 0).all()
        assert np.greater_equal(y, 0).all()
        assert np.greater_equal(z, 0).all()
        self.__x = x
        self.__y = y
        self.__z = z

    def efficiency(self, c: float):
        """the zone space divided by the maximum capacity"""
        return sum(self.__x) * sum(self.__y) * sum(self.__z) / (c * 2) ** 3

    def constraint(self, direction: Vector3) -> float:
        """
        given a direction, find the distance between the center to the surface
        :param direction: the direction (unit vector) the vehicle heads to
        :return: the distance, a.k.a. constraint
        """
        with np.errstate(divide='ignore'):
            constraints = (
                -self.__x[0] / direction.x,
                self.__x[1] / direction.x,
                -self.__y[0] / direction.y,
                self.__y[1] / direction.y,
                -self.__z[0] / direction.z,
                self.__z[1] / direction.z,
            )
        # there should be at most three of them being positive, same for the negative
        # ignore the constraints less than zero because it is the opposite to the direction
        # if here throws an error, it means your direction is a zero vector
        factor = min(i for i in constraints if i >= 0)
        return factor


class Vehicle:
    def __init__(self, v: float, a: float, priority: float, tick: float) -> None:
        """
        a generalized vehicle described by its maximum velocity and acceleration
        :param v: maximum velocity
        :param a: maximum acceleration
        :param priority: the higher priority the more privilege
        :param tick: the unit time
        """
        assert 0 < v
        assert 0 < a
        assert 0 < priority
        assert 0 < tick
        # immutable properties
        self.__v = v
        self.__a = a
        self.__priority = priority  # immutable here, but mutable in practice
        self.__tick = tick
        self.__odometer = 0.0
        # the followings are mutable variables
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
        """the maximum distance this vehicle can reach in a unit time"""
        return self.v * self.tick

    def move(self, zone: Zone, randomness=0.05) -> float:
        """
        the vehicle moves and ensures that it tries its best utilizing the give zone while not exceeding it
        :param zone: the exclusive zone
        :param randomness: indeterministic trajectory, keep it low for vehicles
        :return: moved distance in this unit time
        """
        alpha, beta, gamma = np.random.normal(0.0, self.tick * randomness, size=3)
        # apply randomness to the direction
        rotated = self.direction.t @ spinner(alpha, beta, gamma)
        self.direction = Vector3(*rotated)

        # here is the core trajectory simulation, but the physics is twisted
        # try not to modify this part, it is really fragile
        constraint = zone.constraint(self.direction)
        vi = self.speed

        # find the acceleration the vehicle need to fully utilize the zone
        a = 2 / self.tick ** 2 * (constraint - vi * self.tick)
        # do not decelerate too fast, which leads to negative final velocity
        a = np.clip(a, -min(vi / self.tick, self.a), self.a)

        if a > 0 and (t := abs(self.v - vi) / a) < self.tick:
            # maximum velocity is reached, we calculate them separately
            d1 = vi * t + 1 / 2 * a * t ** 2
            d2 = self.v * (self.tick - t)
            moved, vf = d1 + d2, self.v
        else:
            d = vi * self.tick + 1 / 2 * a * self.tick ** 2
            vf = vi + a * self.tick
            # the clip function gets rid of negative numbers due to float inaccuracy
            # in theory, vf is always greater or equal to zero
            moved, vf = d, np.clip(vf, 0, None)

        self.__odometer += moved

        # update the position and speed
        self.position += self.direction * moved
        self.speed = vf

        return moved


class Human:
    def __init__(self, v: float, theta: float, phi: float, tick: float) -> None:
        """
        a generalized human described by velocity and views
        :param v: velocity
        :param theta: theta of view angle
        :param phi: phi of view angle
        :param tick: the unit time
        """
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
        """
        the human moves with a constant velocity
        :param randomness: indeterministic path, keep it high for humans
        :return: nothing
        """
        alpha, beta, gamma = np.random.normal(0.0, self.tick * randomness, size=3)
        # apply randomness to the direction
        rotated = self.direction.t @ spinner(alpha, beta, gamma)
        self.direction = Vector3(*rotated)

        # update the position
        self.position += self.direction * self.tick * self.v

    def observe(self, vehicle: Vehicle) -> float:
        """
        human take observe and feel the vehicle
        :param vehicle: the target the human is looking at
        :return: the stress of human
        """
        relation = vehicle.position - self.position
        # the calculation is done in polar form for simplicity
        # another twisted calculation, so try not to modify
        h = abs(self.__theta - relation.theta)
        v = abs(self.__phi - relation.phi)
        c = np.pi / 180  # degree to radian factor
        if h > np.pi:
            h = 2 * np.pi - h

        # whether the vehicle is visible to the human
        # view range is about 160 degrees horizontally and 120 degrees vertically
        visible = v < 60 * c and h < 80 * c
        # the distance between the human and the vehicle
        distance = relation.magnitude
        # whether the vehicle is having a level trajectory
        level = abs(vehicle.direction.z) <= 0.2

        # equally weighted
        return 0.0 \
            + 0.25 * (distance < 8 and not level) \
            + 0.25 * (distance < 8 and not visible) \
            + 0.25 * (distance < 3 and vehicle.speed > 0.5) \
            + 0.25 * (distance < 3)
