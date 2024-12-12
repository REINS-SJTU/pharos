import multiprocessing as mp
from typing import List, Any

import numpy as np

from benches.core import Benchmark


def worker(path: str, v: int, limit: int, rep: int, table: List[List[Any]]):
    bench = Benchmark(path, limit, rep)
    for h in range(7):
        table[v - 1][h] = bench.repetition(v, h)


def main(path: str, /, limit=900, rep=100):
    with mp.Manager() as manager:
        table = manager.list([manager.list([None for _ in range(7)]) for _ in range(7)])
        workers = [mp.Process(target=worker, args=(path, v, limit, rep, table)) for v in range(1, 8)]

        for each in workers:
            each.start()
        for each in workers:
            each.join()

        result = np.array(table)
    return result
