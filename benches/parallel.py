import multiprocessing as mp
from typing import List, Any

import numpy as np

from benches.core import Benchmark


def worker(path: str, v: int, table: List[List[Any]]):
    bench = Benchmark(path)
    for h in range(7):
        table[v - 1][h] = bench.repetition(v, h)


def main(path: str):
    with mp.Manager() as manager:
        table = manager.list([manager.list([None for _ in range(7)]) for _ in range(7)])
        workers = [mp.Process(target=worker, args=(path, v, table)) for v in range(1, 8)]

        for each in workers:
            each.start()
        for each in workers:
            each.join()

        result = np.array(table)
    return result
