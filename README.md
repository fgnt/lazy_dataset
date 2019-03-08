# lazy_dataset

[![Build Status](https://travis-ci.org/fgnt/lazy_dataset.svg?branch=master)](https://travis-ci.org/fgnt/lazy_dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgnt/lazy_dataset/blob/master/LICENSE)

Lazy_dataset is a helper to deal with large datasets that do not fit into memory.
It allows to define transformations that are applied lazy,
(e.g. a mapping function to read data from HDD). When someone iterates over the dataset all
transformations are applied.

Supported transformations:
 - `dataset.map(map_fn)`: Apply the function `map_fn` to each example ([builtins.map](https://docs.python.org/3/library/functions.html#map))
 - `dataset[2]`: Get example at index `2`.
 - `dataset['example_id']` Get that example that has the example id `'example_id'`.
 - `dataset[10:20]`: Get a sub dataset that contains only the examples in the slice 10 to 20.
 - `dataset.filter(filter_fn, lazy=True)` Drops examples where `filter_fn(example)` is false ([builtins.filter](https://docs.python.org/3/library/functions.html#filter)).
 - `dataset.concatenate(*others)`: Concaternates two or more datasets ([numpy.concatenate](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.concatenate.html))
 - `dataset.shuffle(reshuffle=False)`: Shuffles the dataset. When `reshuffle` is `True` it shuffles each time when you iterate over the data.
 - `dataset.tile(reps, shuffle=False)`: Repeats the dataset `reps` times and concatenates it ([numpy.tile](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html))
 - `dataset.groupby(group_fn)`: Groups examples together. In contrast to `itertools.groupby` a sort is not nessesary, like in pandas ([itertools.groupby](https://docs.python.org/3/library/itertools.html#itertools.groupby), [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html))
 - `dataset.sort(key_fn, sort_fn=sorted)`: Sorts the examples depending on the values `key_fn(example)` ([list.sort](https://docs.python.org/3/library/stdtypes.html#list.sort))
 - `dataset.batch(batch_size, drop_last=False)`: Batches `batch_size` examples together as a list. Usually followed by a map ([tensorflow.data.Dataset.batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch))
 - `dataset.random_choice()`: Get a random example ([numpy.random.choice](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html))
 - ...


```python
>>> from IPython.lib.pretty import pprint
>>> import lazy_dataset
>>> examples = {
...     'example_id_1': {
...         'observation': [1, 2, 3],
...         'label': 1,
...     },
...     'example_id_2': {
...         'observation': [4, 5, 6],
...         'label': 2,
...     },
...     'example_id_3': {
...         'observation': [7, 8, 9],
...         'label': 3,
...     },
... }
>>> for example_id, example in examples.items():
...     example['example_id'] = example_id
>>> ds = lazy_dataset.new(examples)
>>> ds
  DictDataset(len=3)
MapDataset(_pickle.loads)
>>> ds.keys()
('example_id_1', 'example_id_2', 'example_id_3')
>>> for example in ds:
...     print(example)
{'observation': [1, 2, 3], 'label': 1, 'example_id': 'example_id_1'}
{'observation': [4, 5, 6], 'label': 2, 'example_id': 'example_id_2'}
{'observation': [7, 8, 9], 'label': 3, 'example_id': 'example_id_3'}
>>> def transform(example):
...     example['label'] *= 10
...     return example
>>> ds = ds.map(transform)
>>> for example in ds:
...     print(example)
{'observation': [1, 2, 3], 'label': 10, 'example_id': 'example_id_1'}
{'observation': [4, 5, 6], 'label': 20, 'example_id': 'example_id_2'}
{'observation': [7, 8, 9], 'label': 30, 'example_id': 'example_id_3'}
>>> ds = ds.filter(lambda example: example['label'] > 15)
>>> for example in ds:
...     print(example)
{'observation': [4, 5, 6], 'label': 20, 'example_id': 'example_id_2'}
{'observation': [7, 8, 9], 'label': 30, 'example_id': 'example_id_3'}
>>> ds['example_id_2']
{'observation': [4, 5, 6], 'label': 20, 'example_id': 'example_id_2'}
>>> ds
      DictDataset(len=3)
    MapDataset(_pickle.loads)
  MapDataset(<function transform at 0x7ff74efb6620>)
FilterDataset(<function <lambda> at 0x7ff74efb67b8>)
```


## Installation

Install it directly with Pip, if you just want to use it:

```bash
pip install lazy_dataset
```

If you want to make changes or want the most recent version: Clone the repository and install it as follows:

```bash
git clone https://github.com/fgnt/lazy_dataset.git
cd lazy_dataset
pip install --editable .
```
