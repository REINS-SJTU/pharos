# The MARL Environment

Here are the environment implementation details. Refer to [solution.md](solution.md) for the big picture.

## Initialization

During the initialization or reset stage, it generates all the vehicles and humans. Regardless of the scenario type, it always regenerates the maximum numbers of vehicles and humans and then shuts down the redundant ones. The unit time used is 0.1s.

### Vehicle

All vehicles have the following initial states:

- about 10m away from the origin 
- velocity is about 7m/s towards the origin
- random priority in decimal range from 1 to 3
- maximum velocity at 14m/s
- maximum acceleration at 7m/s^2

It is designed such that:

- the vehicles will crash into each other if nothing stops them, i.e. extreme case
- the vehicles have enough room to slow down before reaching the origin

### Human

All humans have the following initial states:

- about less than 7m away from the origin
- velocity is about 0.5m/s towards a random direction
- random eye direction

It is designed such that:

- it is very likely that vehicles will have to pass around humans
- not too close to vehicle initial position, which can cause inevitable crashes

## Steps

For each step in an episode, the environment does three things. Firstly, all vehicles move based on the given exclusive zone, and humans move as normal. Secondly, every online vehicle makes an observation and also checks if it has crashed. The crashed ones will be temporarily stored in a list instead of being set to offline immediately. Going offline also implies the termination of this agent in this episode. The vehicle will stop after it successfully moves 20m. Lastly, when all vehicles finish observing, the offline flags are populated back.

### Rewards

The base reward is the distance that the vehicle moves within that unit time. Additionally, a small extra reward for a large exclusive zone is also given.

Every vehicle also collects the stress of each human caused by itself. The stress value will reduce the base reward. When two or more vehicles are close to one another and moving at high speed, the base reward is deducted. When the vehicle is close to a human and moving at high speed, the base reward is also deducted. These three deductions are cumulative.

If a vehicle receives a warning, the base reward is forfeited and replaced by a punishment that is proportional to its speed. This forces the vehicle to slow down in order to get rid of the punishment. A warning may occur in two situations: when the vehicle attempts to move while being close to another vehicle that has a higher priority or when the vehicle moves too quickly while being close to a human.

If a vehicle crashes into something, the base reward is forfeited and replaced by a severe punishment. However, if a vehicle successfully moves 20 meters, the base reward is also forfeited and replaced by a substantial reward.
