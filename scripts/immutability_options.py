# This is an integration test for the immutability functionality
from __future__ import annotations
import json
import os
import pickle
import sys
import time
import paderbox
from paderbox.io.cache import url_to_local_path
from collections import defaultdict
from typing import Any
import lazy_dataset
import numpy as np
import psutil
import torch
from tabulate import tabulate


# Download from https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_train2017.json
def create_coco() -> list[Any]:
    json_path = url_to_local_path("https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_train2017.json")
    with open(json_path) as f:
        obj = json.load(f)
        return obj["annotations"]



def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res['rss'] += mmap.rss
        res['pss'] += mmap.pss
        res['uss'] += mmap.private_clean + mmap.private_dirty
        res['shared'] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith('/'):  # looks like a file path
            res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
    return res


class MemoryMonitor():
    """Class used to monitor the memory usage of processes"""

    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)


def read_sample(x):
    """
    A function that is supposed to read object x, incrementing its refcount.
    This mimics what a real dataloader would do."""
    if sys.version_info >= (3, 10, 6):
        """Before this version, pickle does not increment refcount. This is a bug that's
        fixed in https://github.com/python/cpython/pull/92931.     """
        return pickle.dumps(x)
    else:
        import msgpack
        return msgpack.dumps(x)


class DatasetFromList(torch.utils.data.Dataset):
    def __init__(self, lst):
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx: int):
        return self.lst[idx]


def worker(_, dataset: torch.utils.data.Dataset):
    while True:
        for sample in dataset:
            # read the data, with a fake latency
            time.sleep(0.000001)
            result = read_sample(sample)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    monitor = MemoryMonitor()
    immutable_warranty = "pickle"  # copy  pickle  wu
    ds = lazy_dataset.new(create_coco(), immutable_warranty=immutable_warranty)
    print(monitor.table())

    memory = np.zeros((3, 100))
    ctx = torch.multiprocessing.start_processes(
        worker, (ds,), nprocs=4, join=False,
        daemon=True, start_method='fork')
    [monitor.add_pid(pid) for pid in ctx.pids()]

    try:
        for k in range(0, 100):
            print(monitor.table())
            main_pid = os.getpid()
            memory[0, k] = sum([v["uss"] for i, v in monitor.data.items()]) / 1024.0 ** 2
            memory[1, k] = sum([v["pss"] for i, v in monitor.data.items()]) / 1024.0 ** 2
            memory[2, k] = np.mean([v["shared"] for i, v in monitor.data.items()]) / 1024.0 ** 2
            time.sleep(1)
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.set_title("Memory usage of 4 workers")
        axis.plot(memory[0], color="orange", label="USS")
        axis.plot(memory[1], color="b", label="PSS")
        axis.plot(memory[2], color="g", label="Shared")
        axis.set_xlabel("Times (s)")
        axis.legend()
        axis.set_ylabel("Memory usage (MB)")
        # plt.savefig(f"/net/vol/deegen/SHK/Lazy_dataset_test/{immutable_warranty}.svg", format="svg")#, dpi=600)
        plt.show()
    finally:
        ctx.join()
