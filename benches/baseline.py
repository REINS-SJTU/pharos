from typing import Generator, List, Tuple, Any

import numpy as np

from environ.components import Zone
from environ.core import Environ


def baseline(c=1.4, *, limit=900, rep=100) -> np.ndarray:
    """
    fixed size zone for all time, i.e. free fly
    :param c: constraint of the zone
    :param limit: episode length, exceeding this limit leads to failure of episode
    :param rep: repetition of each episode
    :return: numpy array of all data
    """
    bench = Baseline(c, limit, rep)
    result: List[List[Any]] = [[None for _ in range(7)] for _ in range(7)]
    for v in range(1, 8):
        for h in range(7):
            result[v - 1][h] = bench.repetition(v, h)
    return np.array(result)


class Baseline:
    def __init__(self, c: float, limit: int, rep: int):
        """
        init a baseline runner
        :param c: constraint of the zone
        :param limit: episode length, exceeding this limit leads to failure of episode
        :param rep: repetition of each episode
        """
        self.c = c
        self.limit = limit
        self.rep = rep

    def predict(self) -> Generator[Zone, None, None]:
        """
        return a random exclusive zone
        :return: a generator that yields predicted zones
        """
        while True:
            yield Zone((self.c, self.c), (self.c, self.c), (self.c, self.c))

    def onetime(self, vehicles: int, humans: int) -> Tuple[int, int, bool]:
        """
        run the episode of the given scenario
        :param vehicles: number of vehicles
        :param humans: number of humans
        :return: a tuple of crashes, steps, and done
        """
        assert 1 <= vehicles <= 7
        assert 0 <= humans <= 6

        scenario = Environ()
        obs = []
        dones = []
        crashes = 0

        for info in scenario.reset(vehicles, humans):
            obs.append(info)
            dones.append(False)

        for step in range(self.limit):
            zones = self.predict()
            for i, (info, _, crash, done) in enumerate(scenario.step(zones)):
                obs[i] = info
                dones[i] = done
                crashes += crash

            if all(dones):
                return crashes, step, True
        else:
            return crashes, self.limit, False

    def repetition(self, vehicles: int, humans: int) -> List[Tuple[int, int, bool]]:
        """
        run the episode of the given scenario multiple times
        :param vehicles: number of vehicles
        :param humans: number of humans
        :return: a list of all episode results
        """
        assert 1 <= vehicles <= 7
        assert 0 <= humans <= 6

        result = []
        for _ in range(self.rep):
            data = self.onetime(vehicles, humans)
            result.append(data)
        return result
