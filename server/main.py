import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


from contextlib import asynccontextmanager

import gymnasium as gym
import numpy as np
import torch as th
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mappo.algorithms.algorithm.r_actor_critic import R_Actor
from mappo.config import get_config

from pydantic import BaseModel, Field, RootModel

models: dict[str, R_Actor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "results/environ/all/mappo/check/run3/models/actor.pt"
    model = R_Actor(
        get_config().parse_known_args()[0],
        gym.spaces.Box(-np.inf, np.inf, [77], dtype=np.float32),
        gym.spaces.Box(0, 1.4, [6], dtype=np.float32),
    )

    policy_actor_state_dict = th.load(model_path, weights_only=True)
    model.load_state_dict(policy_actor_state_dict)

    models["mappo"] = model
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Observation(RootModel):
    root: list[float] = Field(min_length=77, max_length=77)


class ModelInput(BaseModel):
    model_name: str = "mappo"
    obs: list[Observation]


class ZoneSchema(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


class ModelOutput(BaseModel):
    zones: list[ZoneSchema]


@app.post("/predict")
def model_predict(data: ModelInput) -> list[ZoneSchema]:
    model = models[data.model_name]

    zones: list[ZoneSchema] = []
    for each in data.obs:
        # np.zeros(0) are used to fill rnn states which is not used
        action, _, _ = model(
            np.array(each.model_dump()), np.zeros(0), np.zeros(0), deterministic=True
        )
        action = action.detach().cpu().numpy()[0, 0, 0]
        action = 0.7 * (np.tanh(action) + 1)
        zones.append(ZoneSchema(x=action[:2], y=action[2:4], z=action[4:6]))

    return zones
