# Vehicle Exclusive Zone: MARL Solution

**Core Issue**: How to ensure all vehicles operating safely without knowing their trajectory plans by adjusting their exclusive zones.

**More to Consider**: How to reduce human fear caused by vehicles, and how to adjust vehicle priority.

## Details

Reinforcement learning needs an environment for the agents to interact with. Agents will get rewards and punishments, and the ultimate goal is to maximize the reward. For this particular problem, the exclusive zones for each vehicle are regarded as the agents. Since there are multiple vehicles, multiple agents are needed to predict the exclusive zones for all of them. The environment is a three-dimensional space that includes vehicles and humans, and vehicles react based on their exclusive zones. The whole environment can be compared to a school, where agents are teachers, and each of them is responsible for a student.

**Note**: The vehicles should not be treated as agents because Pharos can only set restrictions on them instead of directly controlling them. Third-party companies have full control over the navigation and utilization of the exclusive zone.

### Dilemma of Causality

**Issue One**: Third parties need to implement a navigation system that follows the Pharos protocol.

**Issue Two**：The agents need to learn the policies based on how vehicles react to exclusive zones.

The solution chooses to use pure physics to simulate the former to solve the latter. It is known that a vehicle must have a target direction and try to reach there as soon as possible. Then we can make the following assumptions:

- Trajectory is not fixed, but very likely moving forwards
- Never going backwards, but slowing down and stopping are allowed
- Utilize the exclusive zone as much as possible, but not exceeding it

### MARL Design

Algorithm: All agents should share the same model, so we only consider MAPPO. Single-agent reinforcement learning is not considered because the problem is cooperation scenario.

Observation Space: It consists of several boxes. Some of them are for technical data of the current vehicle. Some of them are for data of surrounding vehicles. Some of them are for data of surrounding humans. The boxes can also be filled with zeros for any unavailable data.

Action Space: It refers to the restrictions of the exclusive zones in different directions. When its dimension is infinitely high, the zone is perfect. In practice, we just select a reasonable number of dimensions.

## Expectation

We will obtain the actor model. The input is the data of the current vehicle and its surrounding information, and the output is the exclusive zone. After all the calculations are done, Pharos adjusts conflicted exclusive zones and removes obstacles. Then the data is ready to be sent.
