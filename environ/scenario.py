from typing import Generator, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from environ.components import Vehicle, Zone, Human
from environ.utils import Spot


class Scenario:
    def __init__(self) -> None:
        self.vehicles: list[Vehicle] = []
        self.humans: list[Human] = []
        self.__tick = 0.1
        self.__elapsed = 0

    @property
    def tick(self) -> float:
        return self.__tick

    @property
    def elapsed(self) -> float:
        return self.__elapsed

    def step(self, zones: Generator[Zone, None, None]) -> Generator[Tuple[list[float], float, bool], None, None]:
        self.__elapsed += 1

        # everything steps forward, and collect base reward of all vehicles
        rewards = []
        for vehicle, zone in zip(self.vehicles, zones):
            # offline vehicles do not have reward
            if vehicle.offline:
                rewards.append(0.0)
                continue

            # base reward is the distance it moved in this step
            # further adjustments will be applied to it
            moved = vehicle.move(zone)
            # give some extra reward for high spatial efficiency
            moved *= 0.95 + 0.05 * zone.efficiency(vehicle.boundary)
            rewards.append(moved)

        for human in self.humans:
            human.move()

        # all observations are locked here, back population of offline flags is a late update
        offline = []
        for i, vehicle in enumerate(self.vehicles):
            # early yield if this vehicle is already offline
            if vehicle.offline:
                dim = 5 + 8 * (len(self.vehicles) - 1) + 4 * len(self.humans)
                yield [0.0] * dim, rewards[i], True
                continue

            obs = [*vehicle.direction.t, vehicle.speed, vehicle.priority]
            reward = rewards[i]
            # occurs when vehicle safely finished its journey
            done = False
            # occurs when one vehicle hits a human or another vehicle
            crashed = False
            # occurs before bad things happened
            warned = False

            if vehicle.odometer >= 20:
                offline.append(i)
                reward = 10.0
                done = True

            # observe other vehicles
            for other in self.vehicles[:i] + self.vehicles[i + 1:]:
                if other.offline:
                    obs.extend([0.0] * 8)
                    continue

                displacement = other.position - vehicle.position
                info = *displacement.t, *other.direction.t, other.speed, other.priority
                obs.extend(info)

                if warned or crashed:
                    continue

                distance = displacement.magnitude
                # too close
                if distance < 1:
                    offline.append(i)
                    crashed = True
                # encourage vehicles with high priority to move, and the others to wait
                elif distance < 5 and vehicle.priority < other.priority and vehicle.speed > 0:
                    warned = True
                # encourage conservative actions when the surrounding gets complicated
                # cumulated with observation of humans
                elif distance < 10 and vehicle.speed > 5 and other.speed > 5:
                    reward *= 0.9 + 0.01 * distance

            # observe all humans
            for human in self.humans:
                if human.offline:
                    obs.extend([0.0] * 4)
                    continue

                displacement = human.position - vehicle.position
                stress = human.observe(vehicle)
                info = *displacement.t, stress
                obs.extend(info)

                if warned or crashed:
                    continue

                # reduce reward if the human is stressful
                reward *= 0.9 + 0.1 * (1 - stress)

                distance = displacement.magnitude
                # too close
                # crash only if moving, otherwise, it's human's fault
                if distance < 1 < vehicle.speed:
                    offline.append(i)
                    crashed = True
                # a warned of hitting a human
                elif distance < 5 and vehicle.speed > 0.5:
                    warned = True
                # encourage conservative actions when the surrounding gets complicated
                # cumulated with observation of vehicles
                elif distance < 10 and vehicle.speed > 5:
                    reward *= 0.9 + 0.01 * distance

            if crashed:
                yield obs, -10.0, True
            elif warned:
                yield obs, -vehicle.speed / vehicle.v, done
            else:
                yield obs, reward, done

        # late update offline vehicles
        for i in offline:
            self.vehicles[i].offline = True

    def reset(self, vehicles=7, humans=6) -> Generator[List[float], None, None]:
        assert 1 <= vehicles <= 7
        assert 0 <= humans <= 6
        self.__elapsed = 0

        self.vehicles.clear()
        for _ in range(7):
            start = Spot.normal(10, 0.1)
            priority = np.random.uniform(1, 3)
            vehicle = Vehicle(14, 7, priority, self.tick)
            vehicle.position = start
            vehicle.direction = -start.normalized
            vehicle.speed = np.random.normal(7, 0.1)
            self.vehicles.append(vehicle)

        for i in range(7 - vehicles):
            self.vehicles[i].offline = True
        np.random.shuffle(self.vehicles)

        self.humans.clear()
        for _ in range(6):
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(0, np.pi)
            human = Human(1, theta, phi, self.tick)
            human.position = Spot.uniform(0, 7)
            human.direction = Spot.uniform(0, 1)
            self.humans.append(human)

        for i in range(6 - humans):
            self.humans[i].offline = True
        np.random.shuffle(self.humans)

        for i, vehicle in enumerate(self.vehicles):
            obs = [*vehicle.direction.t, vehicle.speed, vehicle.priority]

            for other in self.vehicles[:i] + self.vehicles[i + 1:]:
                if other.offline:
                    obs.extend([0.0] * 8)
                    continue
                displacement = other.position - vehicle.position
                info = *displacement.t, *other.direction.t, other.speed, other.priority
                obs.extend(info)

            for human in self.humans:
                if human.offline:
                    obs.extend([0.0] * 4)
                    continue
                displacement = human.position - vehicle.position
                stress = human.observe(vehicle)
                info = *displacement.t, stress
                obs.extend(info)

            yield obs

    def render(self, vehicle=True, human=True) -> None:
        ax = plt.subplot(projection='3d')
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-10, 10)
        ax.scatter(0, 0, 0, c='black')

        var = np.tanh(self.elapsed / 60)

        if human:
            for human in self.humans:
                ax.scatter(*human.position.t, c=[[0.5, 0.5, 0.5]], s=2)

        if vehicle:
            for vehicle in self.vehicles:
                ax.scatter(*vehicle.position.t, c=[[var, 0.0, 0.0]], s=2)

    @staticmethod
    def demo():
        def helper():
            while True:
                c = (0.7, 0.7)
                yield Zone(c, c, c)

        scenario = Scenario()

        for _ in scenario.reset():
            pass
        scenario.render()

        for _ in range(50):
            zones = helper()
            dones = []
            for _, _, done in scenario.step(zones):
                dones.append(done)
            scenario.render()

            if all(dones):
                break

        plt.show()
