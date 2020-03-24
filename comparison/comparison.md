# Comparison with PyTorch's DataLoader

```python
import lazy_dataset
from torch.utils import data as torch_data
from torch.utils.data import DataLoader as TorchDataLoader
```

| Example | Feature | `lazy_dataset` | `TorchDataLoader` |
| ------- | ------- | :------------: | :---------------: |
|    1    | Dataset class | `lazy_datsaset.Dataset`| `torch_data.Dataset` |
|    2    | Batching | yes | yes |
|    2    | Collate batch | manual | automatic |
|    3    | Shuffle | yes | yes |
|    4    | Multi-process data loading | yes | yes (built-in shared memory GPU) |
|    4    | Multi-thread data loading | yes (default) | no |
|    see [README](../README.md)    | Filter (builtins.filter) | yes | no |
|    5    | Sort by key | yes | no |
|    5    | Sort by value | yes | no |
|    6    | Draw random example | yes | no |
|    7    | Unbatch | yes | no |

## Examples
1. Dataset
    ```python
    >>> examples = {
    ...    'ex_1': {
    ...        'example_id': 'ex_1',
    ...        'observation': [1, 2, 3],
    ...        'label': 1
    ...    },
    ...    'ex_2': {
    ...        'example_id': 'ex_2',
    ...        'observation': [4, 5, 6],
    ...        'label': 2
    ...    }
    ... }

    >>> class TorchDictDataset(torch_data.Dataset):
    ...
    ...    def __init__(self, examples):
    ...        self.examples = examples
    ...
    ...    def __len__(self):
    ...        return len(self.examples)
    ...
    ...    def __getitem__(self, idx):
    ...        idx_to_key = list(self.examples.keys())[idx]
    ...        return self.examples[idx_to_key]

    >>> torch_dict_ds = TorchDictDataset(examples)
    >>> print(torch_dict_ds[0])
    {'example_id': 'ex_1', 'observation': [1, 2, 3], 'label': 1}

    >>> dict_ds = lazy_dataset.from_dict(examples)
    >>> print(dict_ds[0])
    {'example_id': 'ex_1', 'observation': [1, 2, 3], 'label': 1}
    ```
2. Batching & Collate
    ```python
    >>> data_loader = TorchDataLoader(
    ...     torch_dict_ds, batch_size=2, collate_fn=None
    )
    >>> next(iter(data_loader))
    {'example_id': ['ex_1', 'ex_2'],
     'observation': [tensor([1, 4]), tensor([2, 5]), tensor([3, 6])],
     'label': tensor([1, 2])}

    >>> dict_ds = dict_ds.batch(2)
    >>> next(iter(dict_ds))
    [{'example_id': 'ex_1', 'observation': [1, 2, 3], 'label': 1},
     {'example_id': 'ex_2', 'observation': [4, 5, 6], 'label': 2}]
    >>> def collate_fn(batch):
    ...     batched_values = list(zip(*[ex.values() for ex in batch]))
    ...     return {k: v for k, v in zip(batch[0].keys(), batched_values)}
    >>> dict_ds = dict_ds.map(collate_fn)
    >>> next(iter(dict_ds))
    {'example_id': ('ex_1', 'ex_2'),
     'observation': ([1, 2, 3], [4, 5, 6]),
     'label': (1, 2)}
    ```
3. Shuffle
    ```python
    >>> data_loader = TorchDataLoader(torch_dict_ds, shuffle=True)
    >>> dict_ds = dict_ds.shuffle()
    ```
4. Parallel data loading
    ```python
    >>> data_loader = TorchDataLoader(torch_dict_ds, num_workers=4)
    # default: use threading backend
    >>> dict_ds_t = dict_ds.prefetch(num_workers=4, buffer_size=8, backend='t')
    # multi-process backend
    >>> dict_ds_mp = dict_ds.prefetch(num_workers=4, buffer_size=8, backend='mp')
    ```
5. Sort
  ```python
  >>> examples = {
  ...     'a': {
  ...         'observation': [1, 2, 3],
  ...         'label': 2
  ...     },
  ...     'b': {
  ...         'observation': [4, 5, 6],
  ...         'label': 3
  ...     },
  ...     'c': {
  ...         'observation': [7, 8, 9],
  ...         'label': 1
  ...     }
  ... }

  >>> ds = lazy_dataset.from_dict(examples)
  >>> for key, example in ds.items():
  ...     print(key, example)
  a {'observation': [1, 2, 3], 'label': 2}
  b {'observation': [4, 5, 6], 'label': 3}
  c {'observation': [7, 8, 9], 'label': 1}
  # Sort by value
  >>> ds_sorted = ds.sort(lambda ex: ex['label'])
  >>> for key, example in ds_sorted.items():
  ...     print(key, example)
  c {'observation': [7, 8, 9], 'label': 1}
  a {'observation': [1, 2, 3], 'label': 2}
  b {'observation': [4, 5, 6], 'label': 3}
  # Sort by key
  >>> for key, example in (ds_sorted.sort()).items():
  ...     print(key, example)
  a {'observation': [1, 2, 3], 'label': 2}
  b {'observation': [4, 5, 6], 'label': 3}
  c {'observation': [7, 8, 9], 'label': 1}
  ```
6. Draw random example
  ```python
  >>> import numpy as np
  >>> rng_state = np.random.RandomState(0)
  >>> examples = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
  >>> ds = lazy_dataset.from_dict(examples)
  >>> ds.random_choice(rng_state=rng_state)
  3
  >>> print(ds.random_choice(7, rng_state=rng_state, replace=True))
  SliceDataset([3 3 3 1 3 2 4])
  ```
7. Unbatch (with local shuffle)
  ```python
  >>> examples = {'a': [1, 2], 'b': [3, 4]}
  >>> ds = lazy_dataset.from_dict(examples)
  >>> list(ds)
  [[1, 2], [3, 4]]
  >>> list(ds.unbatch())
  [1, 2, 3, 4]
  >>> list(ds.unbatch().shuffle(reshuffle=True, buffer_size=4))
  [3, 1, 2, 4]
  ```

## Throughput
To compare the throughput (loaded examples per second) with PyTorch's DataLoader, the data pipeline was designed to consider two kinds of load:
* I/O load: First, audio sequences from the [LibriSpeech corpus](http://www.openslr.org/12/) are loaded into RAM.
* CPU load: Given the audio sequences, STFT spectrograms (FFT size=512, shift=128) are computed.

Then, the spectrograms are shuffled, batched into small mini-batches and padded to the same sequence length to yield tensors of shape B x T x F.
This corresponds to a common data pipeline which we are using for our research experiments.

The throughput is calculated for an iteration through the `train_clean_100` dataset which contains 28539 audio sequences.
Each dataset iteration is repeated ten times and the average throughput is reported.
The throughput is plotted against the number of workers used for data fetching.
`Number Workers = 0` means that no sub-processes / threads are spawned and all
data is loaded in the main thread.

### Environment
```
OS: Ubuntu 18.04.4 LTS
Python version: 3.6.8
torch version: 1.0.0
torch.version.cuda: 9.0.176
lazy_dataset version: 0.0.6
```

### CPU

In a first experiment, the data is only loaded onto the CPU.

CPU specifications:
```bash
$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              8
On-line CPU(s) list: 0-7
Thread(s) per core:  2
Core(s) per socket:  4
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               158
Model name:          Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz
Stepping:            9
CPU MHz:             3562.482
CPU max MHz:         4100.0000
CPU min MHz:         800.0000
BogoMIPS:            7392.00
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            8192K
NUMA node0 CPU(s):   0-7
```
![Throughput CPU](
  throughput_cpu_bs16.png "Throughput when loading onto CPU"
)

### GPU

In a second experiment, the data is additionally transferred to the GPU.
The `pin_memory` flag puts the tensors in pinned memory during multi-process data loading for fast transfer to the GPU.

CPU specifications:
```bash
$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              8
On-line CPU(s) list: 0-7
Thread(s) per core:  2
Core(s) per socket:  4
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               158
Model name:          Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz
Stepping:            9
CPU MHz:             800.113
CPU max MHz:         4100.0000
CPU min MHz:         800.0000
BogoMIPS:            7392.00
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            8192K
NUMA node0 CPU(s):   0-7
```
GPU specifications:
```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.116                Driver Version: 390.116                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 970     Off  | 00000000:01:00.0  On |                  N/A |
|  0%   39C    P8    17W / 163W |     32MiB /  4041MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
```

![Throughput GPU](
  throughput_gpu_bs32.png "Throughput when loading onto GPU"
)

### Reproduce

The plots were created with following script:
```python
# On CPU: python time_loaders.py
# On GPU: export CUDA_VISIBLE_DEVICES=0; python time_loaders.py --gpu
import time
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import lazy_dataset
import tqdm
import click

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# https://github.com/fgnt/paderbox
from paderbox.utils.nested import flatten
from paderbox.io.audioread import load_audio
from paderbox.transform.module_stft import spectrogram

# https://github.com/fgnt/padertorch
from padertorch.contrib.jensheit.data import Padder

# exclusive to NT group of Paderborn University
from padercontrib.database.librispeech import LibriSpeech


class Prepare:

    def __call__(self, example):
        # IO
        example['audio_data'] = (
            load_audio(flatten(example)['audio_path.observation'])
            .astype('float32')
        )
        # CPU load
        example['spectrogram'] = (
            spectrogram(example['audio_data'], size=512, shift=128)
            .astype(np.float32)
        )
        return example


class ShuffleSampler(Sampler):

    def __init__(self, data_source, seed=None):
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.seed = seed

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        rng = (
            np.random if self.seed is None else np.random.RandomState(self.seed)
        )
        permutation = np.arange(len(self))
        rng.shuffle(permutation)
        return iter(permutation)


def _prepare_lazy_dataset(dataset, num_workers=0, backend='t', batch_size=16):
    as_lazy_dataset = (
        dataset
        .shuffle(rng=np.random.RandomState(0))
        .batch(batch_size, drop_last=False)
        .map(Padder(padding_keys=['spectrogram']))
    )
    if num_workers:
        as_lazy_dataset = as_lazy_dataset.prefetch(
            num_workers, 2 * num_workers, backend=backend
        )
    return as_lazy_dataset


def _prepare_data_loader(
    dataset, num_workers=0, pin_memory=False, batch_size=16
):
    padder = Padder(padding_keys=['spectrogram'])

    def collate_wrapper(batch):
        return padder(batch)

    as_torch_data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # custom collating: pad spectrograms to same length
        collate_fn=collate_wrapper,
        drop_last=False,
        # ensure that shuffling yields the same batches as lazy_dataset
        sampler=ShuffleSampler(dataset, seed=0),
        num_workers=num_workers, pin_memory=pin_memory
    )
    return as_torch_data_loader


def assert_example(
    dataset, batch_size=16, keys=['spectrogram'], pin_memory=False,
    to_gpu=False
):
    example = next(iter(dataset))
    for key in keys:
        assert isinstance(example[key], torch.Tensor), (key, type(example[key]))
        assert example[key].shape[0] == batch_size, (key, example[key].shape)
        assert example[key].is_pinned() == pin_memory, (
            key, example[key].is_pinned(), pin_memory
        )
        if to_gpu:
            x = example[key].to(0)
            assert x.is_cuda is True, (x.device, to_gpu)


def iteration_timing(dataset, size, key='spectrogram', runs=10, to_gpu=False):
    time_per_run = []
    for i in range(runs):
        start = time.time()
        for example in iter(dataset):
            # access data
            _ = example['spectrogram'].to(0 if to_gpu else 'cpu')
        duration = time.time() - start
        logging.info(f'Duration for run {i}: {duration:.2f}s')
        time_per_run.append(duration)
    # throughput: loaded examples per second
    return size // np.median(time_per_run)  # works better for outliers than np.mean


def take_timings(
    dataset, runs=10, num_workers=0, backends=['t'], to_gpu=False, batch_size=16
):
    num_examples = len(dataset)
    throughput_torch_data_loader = []
    if to_gpu:
        pin_memory_runs = [False, True]
    else:
        pin_memory_runs = [False]
    for pin_memory in pin_memory_runs:
        as_torch_data_loader = _prepare_data_loader(
            dataset, num_workers, pin_memory=pin_memory, batch_size=batch_size
        )
        assert_example(
            as_torch_data_loader, batch_size=batch_size, pin_memory=pin_memory,
            to_gpu=to_gpu
        )
        logging.info(
            f'Taking timings for torch.DataLoader (pin_memory={pin_memory})'
        )
        throughput_torch_data_loader.append(iteration_timing(
            as_torch_data_loader, num_examples, runs=runs, to_gpu=to_gpu
        ))

    throughput_lazy_dataset = []
    for backend in backends:
        as_lazy_dataset = _prepare_lazy_dataset(
            dataset, num_workers, backend=backend, batch_size=batch_size
        )
        assert_example(as_lazy_dataset, batch_size=batch_size, to_gpu=to_gpu)
        logging.info(f'Taking timings for lazy_dataset (backend={backend})')
        throughput_lazy_dataset.append(iteration_timing(
            as_lazy_dataset, num_examples, runs=runs, to_gpu=to_gpu
        ))
    return throughput_torch_data_loader, throughput_lazy_dataset


def plot_timings(
    x, t_ld, t_td, num_examples, backends=['t'], to_gpu=False,
    outfile='throughput.png'
):
    if to_gpu:
        pin_memory_runs = [False, True]
    else:
        pin_memory_runs = [False]
    markers_ld = ['o', 'x', '^', 'd']
    markers_td = ['s', '^', 'd']
    for i, backend in enumerate(backends):
        plt.plot(
            x, [t[i] for t in t_ld], marker=markers_ld[i],
            label=f'lazy_dataset, backend={backend}'
        )
    for i, pin_memory in enumerate(pin_memory_runs):
        plt.plot(
            x, [t[i] for t in t_td], marker=markers_td[i],
            label=f'torch.DataLoader, pin_memory={pin_memory}'
        )
    plt.xlabel('Number Workers')
    plt.xticks(x)
    plt.ylabel(r'Throughput (examples per $s$)')
    plt.legend()
    plt.grid()
    logging.info(f'Saved to {outfile}.')
    plt.savefig(outfile)


@click.command()
@click.option('--batch-size', type=int, default=16)
@click.option('--runs', type=int, default=10)
@click.option('--gpu', is_flag=True)
@click.option('--outfile', type=str, default='throughput.png')
def main(batch_size, runs, gpu, outfile):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # store LibriSpeech audio paths in dict
    db = LibriSpeech()
    data = db.get_examples('train_clean_100')
    """
    >>> import pprint
    >>> pprint.pprint(data['103-1240-0000'])
    {'audio_path': {'observation': '/net/db/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac'},
     'gender': 'f',
     'num_samples': 225360,
     'speaker_id': '103-1240',
     'example_id': '103-1240-0000',
     'dataset': 'train_clean_100'}
    """

    dataset = lazy_dataset.from_dict(data)
    dataset = dataset.map(Prepare())
    # cache audio_data, otherwise the first iteration through the data will be
    # slower than the following ones
    for example in tqdm.tqdm(
        iter(dataset.prefetch(num_workers=4, buffer_size=8)),
        total=len(dataset),
        desc=f'Cache audio_data (num_workers=4)'
    ):
        _ = example['audio_data']
    workers_list = [0, 1, 2, 4, 8]
    backends = ['t', 'concurrent_mp']
    timings_lazy_dataset = list()
    timings_torch_data_loader = list()
    for num_workers in workers_list:
        logging.info(
            f'Starting timing measurement for num_workers={num_workers}'
        )
        t_td, t_ld = take_timings(
            dataset, num_workers=num_workers, runs=runs, backends=backends,
            to_gpu=gpu, batch_size=batch_size
        )
        timings_lazy_dataset.append(t_ld)
        timings_torch_data_loader.append(t_td)
    plot_timings(
        workers_list, timings_lazy_dataset, timings_torch_data_loader,
        len(data), backends=backends, to_gpu=gpu, outfile=outfile
    )


if __name__ == '__main__':
    main()

```
