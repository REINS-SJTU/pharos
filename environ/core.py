from typing import Generator, Tuple, List

import numpy as np

from environ.components import Vehicle, Zone, Human
from environ.utils import Spot


class Environ:
    def __init__(self) -> None:
        """
        the environment for the agents to interact with
        """
        self.vehicles: list[Vehicle] = []
        self.humans: list[Human] = []

        # the tick (in seconds) is essential
        # small value leads to precises control and more adventurous actions
        # large value leads to conservative actions, because too many things may happen in a tick
        # make sure it is larger than the sum of all network and computation latencies
        # if a big change is made to it, consider re-designing the whole environment
        self.__tick = 0.1

    @property
    def tick(self) -> float:
        return self.__tick

    def step(self, zones: Generator[Zone, None, None]) -> Generator[Tuple[list[float], float, bool, bool], None, None]:
        """
        the episode steps forward, i.e. all vehicles and humans move for a tick
        :param zones: a zone generator for each vehicle
        :return: yield the observation of a single vehicle, the sequence is fixed for this episode
        """

        rewards = []
        # vehicles steps forward, and collect base reward of all vehicles
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

        # humans steps forward
        for human in self.humans:
            human.move()

        # all observations are locked here, back population of offline flags is a late update
        offline = []
        for i, vehicle in enumerate(self.vehicles):
            # early yield if this vehicle is already offline
            if vehicle.offline:
                dim = 5 + 8 * (len(self.vehicles) - 1) + 4 * len(self.humans)
                yield [0.0] * dim, rewards[i], False, True
                continue

            # init with properties of this vehicle, will be extended
            obs = [*vehicle.direction.t, vehicle.speed, vehicle.priority]
            # the base word
            reward = rewards[i]
            # occurs when vehicle safely finished its journey
            done = False
            # occurs when one vehicle hits a human or another vehicle
            crashed = False
            # occurs before bad things happened
            warned = False

            if vehicle.odometer >= 20:
                offline.append(i)
                done = True

            # observe other vehicles
            for other in self.vehicles[:i] + self.vehicles[i + 1:]:
                if other.offline:
                    obs.extend([0.0] * 8)
                    continue

                displacement = other.position - vehicle.position
                info = *displacement.t, *other.direction.t, other.speed, other.priority
                obs.extend(info)

                if done or warned or crashed:
                    continue

                distance = displacement.magnitude
                # too close
                if distance < 1.0:
                    offline.append(i)
                    crashed = True
                # encourage vehicles with high priority to move, and the others to wait
                elif distance < 5.0 and vehicle.priority < other.priority and vehicle.speed > 0.0:
                    warned = True
                # encourage conservative actions when the surrounding gets complicated
                elif distance < 10.0 and vehicle.speed > 5.0 and other.speed > 5.0:
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

                if done or warned or crashed:
                    continue

                # reduce reward if the human is stressful
                reward *= 0.9 + 0.1 * (1.0 - stress)

                distance = displacement.magnitude
                # crash only if the vehicle moves too fast
                # human will stay away from vehicles, if not, it is human's fault
                if distance < 1.0 < vehicle.speed:
                    offline.append(i)
                    crashed = True
                # a warned of hitting a human
                elif distance < 5 and vehicle.speed > 0.5:
                    warned = True
                # encourage conservative actions when the surrounding gets complicated
                elif distance < 10.0 and vehicle.speed > 5.0:
                    reward *= 0.9 + 0.01 * distance

            # obs, reward, crash, done
            if done:
                yield obs, 10.0, False, True
            elif crashed:
                yield obs, -10.0, True, True
            elif warned:
                yield obs, -vehicle.speed / vehicle.v, False, False
            else:
                yield obs, reward, False, False

        # late update offline vehicles
        for i in offline:
            self.vehicles[i].offline = True

    def reset(self, vehicles=7, humans=6) -> Generator[List[float], None, None]:
        """
        reset all parameters and regenerate all vehicles and humans
        :param vehicles: number of vehicles
        :param humans: number of humans
        :return: yield the observation of a single vehicle, the sequence is fixed for this episode
        """
        assert 1 <= vehicles <= 7
        assert 0 <= humans <= 6

        # reset all
        self.vehicles.clear()
        self.humans.clear()

        # generate vehicles
        for _ in range(7):
            # immutable properties
            start = Spot.normal(10, 0.1)
            priority = np.random.uniform(1, 3)
            vehicle = Vehicle(14, 7, priority, self.tick)
            # mutable properties
            vehicle.position = start
            vehicle.direction = -start.normalized  # point towards the origin
            vehicle.speed = np.random.normal(7, 0.1)
            self.vehicles.append(vehicle)

        # shut down vehicles that are not needed
        for i in range(7 - vehicles):
            self.vehicles[i].offline = True
        np.random.shuffle(self.vehicles)

        # generate humans
        for _ in range(6):
            # immutable properties
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(0, np.pi)
            human = Human(0.5, theta, phi, self.tick)
            # mutable properties
            human.position = Spot.uniform(0, 7)
            human.direction = Spot.at(1)  # random unit vector
            self.humans.append(human)

        # shut down humans that are not needed
        for i in range(6 - humans):
            self.humans[i].offline = True
        np.random.shuffle(self.humans)

        for i, vehicle in enumerate(self.vehicles):
            # init with properties of this vehicle, will be extended
            obs = [*vehicle.direction.t, vehicle.speed, vehicle.priority]

            # observe other vehicles
            for other in self.vehicles[:i] + self.vehicles[i + 1:]:
                if other.offline:
                    obs.extend([0.0] * 8)
                    continue
                displacement = other.position - vehicle.position
                info = *displacement.t, *other.direction.t, other.speed, other.priority
                obs.extend(info)

            # observe all humans
            for human in self.humans:
                if human.offline:
                    obs.extend([0.0] * 4)
                    continue
                displacement = human.position - vehicle.position
                stress = human.observe(vehicle)
                info = *displacement.t, stress
                obs.extend(info)

            yield obs
