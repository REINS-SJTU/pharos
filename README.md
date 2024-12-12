# Pharos

This repository tried to solve the problem of spatial management for all transportation means in a human-robot coexistence environment. In other words, how to keep both vehicles and humans safe without controlling them directly. Multi-agent reinforcement learning is used to approach the solution, and the [mappo](mappo) implementation is a clone of [light_mappo](https://github.com/tinyzqh/light_mappo) with some modifications.

## Installation

Pick a virtual environment and install the dependencies by:

```shell
pip3 install -r requirements.txt
```

## Usage

To train the model with optional [arguments](mappo/config.py):

```shell
python3 train.py
```

## Benchmark

Benchmark procedure is up to the users, but some helper functions are available in module [benches](benches). Refer to [analysis.ipynb](analysis.ipynb) for basic usage and sample analysis.

## License

Distributed under the terms of the [MIT License](LICENSE).
