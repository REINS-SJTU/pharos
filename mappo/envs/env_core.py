import numpy as np

from environ.components import Zone
from environ.scenario import Scenario


class EnvCore(object):
    def __init__(self):
        self.scenario = Scenario()
        self.agent_num = 7
        self.obs_dim = 77
        self.action_dim = 6

    def reset(self):
        sub_agent_obs = []
        vehicles = np.random.randint(1, 8)
        humans = np.random.randint(0, 7)
        for obs in self.scenario.reset(vehicles, humans):
            sub_agent_obs.append(obs)
        return sub_agent_obs

    def step(self, actions):
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        zones = (Zone(i[:2], i[2:4], i[4:6]) for i in actions)
        for obs, reward, done in self.scenario.step(zones):
            sub_agent_obs.append(obs)
            sub_agent_reward.append([reward])
            sub_agent_done.append(done)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
