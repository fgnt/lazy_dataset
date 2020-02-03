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
|    4    | Multi-process data loading | yes | yes |
|    see [README](../README.md)    | Filter (builtins.filter) | yes | no |
|    5    | Sort by key | yes | no |
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
    >>> dict_ds = dict_ds.prefetch(num_workers=4, buffer_size=4)
    ```
5. Sort by key
  ```python
  >>> examples = {
  ...     'ex_1': {
  ...         'example_id': 'ex_1',
  ...         'observation': [1, 2, 3],
  ...         'label': 2
  ...     },
  ...     'ex_2': {
  ...         'example_id': 'ex_2',
  ...         'observation': [4, 5, 6],
  ...         'label': 3
  ...     },
  ...     'ex3': {
  ...         'example_id': 'ex_3',
  ...         'observation': [7, 8, 9],
  ...         'label': 1
  ...     }
  ... }

  >>> ds = lazy_dataset.from_dict(examples)
  >>> for example in ds:
  ...     print(example)
  {'example_id': 'ex_1', 'observation': [1, 2, 3], 'label': 2}
  {'example_id': 'ex_2', 'observation': [4, 5, 6], 'label': 3}
  {'example_id': 'ex_3', 'observation': [7, 8, 9], 'label': 1}
  >>> ds = ds.sort(lambda x: x['label'])
  >>> for example in ds:
  ...     print(example)
  {'example_id': 'ex_3', 'observation': [7, 8, 9], 'label': 1}
  {'example_id': 'ex_1', 'observation': [1, 2, 3], 'label': 2}
  {'example_id': 'ex_2', 'observation': [4, 5, 6], 'label': 3}
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
7. Unbatch
  ```python
  >>> examples = {'a': [1, 2], 'b': [3, 4]}
  >>> ds = lazy_dataset.from_dict(examples)
  >>> list(ds)
  [[1, 2], [3, 4]]
  >>> list(ds.unbatch())
  [1, 2, 3, 4]
  ```

## Throughput
To compare the throughput (loaded examples per second) with PyTorch's DataLoader, the following scenario was chosen:
Audio sequences from the [LibriSpeech corpus](http://www.openslr.org/12/) are loaded into RAM, batched into chunks of 16 sequences and the sequences in each batch are zero-padded to the same length.
The throughput is calculated for a whole iteration of the complete `train_clean_100` dataset which contains 28539 audio sequences.
Each dataset iteration is repeated ten times and the averaged throughput is reported.
In the first experiment, the data is only loaded onto the CPU.
In the second experiment, the data is additionally transferred to the GPU.

Loading onto CPU:  
![Throughput CPU](
  throughput_timings_size-2000_shuffle_librispeech_runs-10_batch-size16.png "Throughput when loading onto CPU"
)

Loading onto GPU:  
![Throughput GPU](
  throughput_timings_size-2000_shuffle_librispeech_runs-10_batch-size16_gpu.png "Throughput when loading onto GPU"
)

GPU throughput is higher because a different machine was used.
The plots were created with following script:
```python
# On CPU: python throughput.py
# On GPU: export CUDA_VISIBLE_DEVICES=0; python throughput.py --gpu
import os
from functools import partial
import time

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
from paderbox.io.audioread import load_audio
from paderbox.utils.nested import flatten

# https://github.com/fgnt/padertorch
import padertorch as pt
from padertorch.contrib.jensheit.data import Padder

# exclusive to NT group of Paderborn University
from padercontrib.database.librispeech import LibriSpeech


class AudioReadMap:

    def __init__(self, key):
        self.key = key

    def __call__(self, example):
        example['audio_data'] = load_audio(flatten(example)[self.key])
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


class WrapDataLoaderToGPU:

    def __init__(self, data_loader, key, device=0, non_blocking=False):
        """
        Wraps a torch DataLoader to yield CUDA tensors during iteration
        """
        self.loader = data_loader
        self.key = key
        self.device = device
        self.non_blocking = non_blocking

    def __iter__(self):
        for example in self.loader:
            if not self.non_blocking:
                example = pt.data.example_to_device(example, device=self.device)
            else:
                example[self.key] = example[self.key].to(
                    device=self.device, non_blocking=True
                )
            yield example


def _prepare_lazy_dataset(dataset, num_workers=0, to_gpu=False):
    as_lazy_dataset = (
        dataset
            .shuffle(rng=np.random.RandomState(0))
            .batch(16, drop_last=False)
            .map(Padder(padding_keys=['audio_data']))
    )
    if num_workers:
        as_lazy_dataset = as_lazy_dataset.prefetch(
            num_workers, 16  # prefetch one batch
        )
    if to_gpu:
        as_lazy_dataset = as_lazy_dataset.map(
            partial(pt.data.example_to_device, device=0)
        )
    return as_lazy_dataset


def _prepare_data_loader(dataset, num_workers=0, pin_memory=False, to_gpu=False):
    as_torch_data_loader = DataLoader(
        dataset,
        batch_size=16,
        # custom collating: pad audio sequences to same length
        collate_fn=Padder(padding_keys=['audio_data']),
        drop_last=False,
        # ensure that shuffling yields the same batches as lazy_dataset
        sampler=ShuffleSampler(dataset, seed=0),
        num_workers=num_workers, pin_memory=pin_memory
    )
    if to_gpu:
        as_torch_data_loader = WrapDataLoaderToGPU(
            as_torch_data_loader, 'audio_data', device=0,
            non_blocking=pin_memory
        )
    return as_torch_data_loader


def assert_example(dataset, batch_size=16, key='audio_data', on_gpu=False):
    example = next(iter(dataset))
    assert isinstance(example[key], torch.Tensor), type(example[key])
    assert example[key].ndimension() >= 2, example[key].shape
    assert example[key].shape[0] == batch_size, example[key].shape
    assert example[key].is_cuda == on_gpu, (example[key].device, on_gpu)


def iteration_timing(
    dataset, size, key='audio_data', runs=10
):
    total_time = 0
    for _ in range(runs):
        start = time.time()
        for example in iter(dataset):
            # access data
            _ = example[key]
        total_time += time.time() - start
    # throughput: loaded examples (audio sequences) per second
    return size * runs // total_time


def take_timings(dataset, runs=10, num_workers=0, to_gpu=False):
    num_examples = len(dataset)
    as_lazy_dataset = _prepare_lazy_dataset(dataset, num_workers, to_gpu=to_gpu)
    as_torch_data_loader = _prepare_data_loader(
        dataset, num_workers, to_gpu=to_gpu
    )
    assert_example(as_lazy_dataset, batch_size=16, on_gpu=to_gpu)
    assert_example(as_torch_data_loader, batch_size=16, on_gpu=to_gpu)

    throughput_lazy_dataset = iteration_timing(
        as_lazy_dataset, num_examples, runs=runs
    )
    throughput_torch_data_loader = iteration_timing(
        as_torch_data_loader, num_examples, runs=runs
    )
    return throughput_lazy_dataset, throughput_torch_data_loader


def plot_timings(x, t_ld, t_td, num_examples):
    plt.plot(x, t_ld, marker='o', label='lazy_dataset')
    plt.plot(
        x, t_td, marker='s', label='torch.DataLoader'
    )
    plt.title(
        f'Batch, pad, shuffle, iterate over {num_examples} sequences\n'
        '(from LibriSpeech train_clean_100)'
    )
    plt.xlabel('Number Workers')
    plt.xticks(x)
    plt.ylabel(r'Throughput (examples per $s$)')
    plt.legend()
    plt.grid()
    plt.savefig('throughput.png')


@click.command()
@click.option('--runs', type=int, default=10)
@click.option('--gpu', is_flag=True)
def main(runs, gpu):
    # store LibriSpeech audio paths in a dict
    db = LibriSpeech()
    data = db.get_examples('train_clean_100')

    dataset = lazy_dataset.from_dict(data)
    dataset = dataset.map(
        AudioReadMap(key='audio_path.observation')
    )
    # cache audio_data, otherwise the first iteration through the data will be
    # slower than the following ones
    for example in tqdm.tqdm(
        iter(dataset.prefetch(num_workers=4, buffer_size=4)),
        total=len(dataset),
        desc=f'Cache audio_data (num_workers=4)'
    ):
        _ = example['audio_data']
    workers_list = [0, 1, 2, 4, 8]
    timings_lazy_dataset = list()
    timings_torch_data_loader = list()
    for num_workers in workers_list:
        t_ld, t_td = take_timings(
            dataset, num_workers=num_workers, runs=runs, to_gpu=gpu
        )
        timings_lazy_dataset.append(t_ld)
        timings_torch_data_loader.append(t_td)
    plot_timings(
        workers_list, timings_lazy_dataset, timings_torch_data_loader, len(data)
    )


if __name__ == '__main__':
    main()
```
