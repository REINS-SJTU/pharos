from typing import Any, Tuple, List

import gymnasium as gym
import numpy as np
import torch as th

from environ.components import Zone
from environ.core import Environ
from mappo.algorithms.algorithm.r_actor_critic import R_Actor
from mappo.config import get_config


class Benchmark:
    def __init__(self, path: str):
        model = R_Actor(
            get_config().parse_known_args()[0],
            gym.spaces.Box(-np.inf, np.inf, [77], dtype=np.float32),
            gym.spaces.Box(0, 1.4, [6], dtype=np.float32),
        )

        policy_actor_state_dict = th.load(path, weights_only=True)
        model.load_state_dict(policy_actor_state_dict)

        self.model = model
        self.limit = 800
        self.rep = 100

    def predict(self, obs: List[Any]):
        for each in obs:
            # np.zeros(0) are used to fill rnn states which is not used
            action, _, _ = self.model(np.array(each), np.zeros(0), np.zeros(0), deterministic=True)
            action = action.detach().cpu().numpy()[0, 0, 0]
            action = 0.7 * (np.tanh(action) + 1)
            yield Zone(action[:2], action[2:4], action[4:6])

    def onetime(self, vehicles: int, humans: int) -> Tuple[int, int, bool]:
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
            zones = self.predict(obs)
            for i, (info, _, crash, done) in enumerate(scenario.step(zones)):
                obs[i] = info
                dones[i] = done
                crashes += crash

            if all(dones):
                return crashes, step, True
        else:
            return crashes, self.limit, False

    def repetition(self, vehicles: int, humans: int) -> Tuple[List[int], List[int], List[bool]]:
        assert 1 <= vehicles <= 7
        assert 0 <= humans <= 6

        crashes = []
        steps = []
        dones = []
        for _ in range(self.rep):
            crash, step, done = self.onetime(vehicles, humans)
            crashes.append(crash)
            steps.append(step)
            dones.append(done)
        return crashes, steps, dones
