import pickle
import logging
import numbers
import textwrap
import operator
from copy import deepcopy
import itertools
import functools
import collections
from pathlib import Path
import time
import datetime

import numpy as np
from typing import Optional, Union, Any, List, Dict, Tuple, Generator

LOG = logging.getLogger('lazy_dataset')


def _get_serialize_and_deserialize(immutable_warranty):
    if immutable_warranty == 'pickle':
        return pickle.dumps, pickle.loads
    elif immutable_warranty == 'copy':
        return lambda x: x, deepcopy
    else:
        raise ValueError(immutable_warranty)


def new(
        examples: Union[list, dict, 'Dataset'],
        immutable_warranty: str = 'pickle',
        name: str = None,
):
    """
    Creates a new dataset from data in `examples`. `examples` can be a `list`
    or a `dict`.

    Args:
        examples: The data to create a new dataset from
        immutable_warranty: How to ensure immutability. Available options are
            'pickle' and 'copy'.
        name: An optional name for the dataset. Only affects the representer.

    Returns:
        The `Dataset` created from `examples`

    Examples:
        Create a dataset from a dict:

        >>> import lazy_dataset
        >>> ds = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3}, name='MyDataset')
        >>> ds
          DictDataset(name='MyDataset', len=3)
        MapDataset(_pickle.loads)
        >>> ds.keys()
        ('a', 'b', 'c')
        >>> list(ds)
        [1, 2, 3]
        >>> ds = ds.map(lambda example: example * 2)
        >>> list(ds)
        [2, 4, 6]
        >>> ds = ds.filter(lambda example: example > 2)
        >>> list(ds)
        [4, 6]
        >>> ds  # doctest: +ELLIPSIS
              DictDataset(name='MyDataset', len=3)
            MapDataset(_pickle.loads)
          MapDataset(<function <lambda> at ...>)
        FilterDataset(<function <lambda> at ...>)

        Create a dataset from a list:

        >>> ds = lazy_dataset.new([1, 2, 3, 4, 5])
        >>> list(ds)
        [1, 2, 3, 4, 5]
        >>> lazy_dataset.new(ds)
          ListDataset(len=5)
        MapDataset(_pickle.loads)

    """
    if isinstance(examples, dict):
        dataset = from_dict(
            examples, immutable_warranty=immutable_warranty, name=name)
    elif isinstance(examples, (tuple, list)):
        dataset = from_list(
            examples, immutable_warranty=immutable_warranty, name=name)
    elif isinstance(examples, Dataset):
        dataset = from_dataset(
            examples, immutable_warranty=immutable_warranty, name=name)
    else:
        raise TypeError(type(examples), examples)
    return dataset


def from_dict(
        examples: dict,
        immutable_warranty: str = 'pickle',
        name: str = None,
):
    serialize, deserialize = _get_serialize_and_deserialize(immutable_warranty)
    examples = {k: serialize(v) for k, v in examples.items()}
    return DictDataset(examples, name=name).map(deserialize)


def from_list(
        examples: [list, tuple],
        immutable_warranty: str = 'pickle',
        name: str = None,
):
    assert isinstance(examples, (tuple, list)), examples
    serialize, deserialize = _get_serialize_and_deserialize(immutable_warranty)
    examples = list(map(serialize, examples))
    return ListDataset(examples, name=name).map(deserialize)


def from_dataset(
        examples: 'Dataset',
        immutable_warranty: str = 'pickle',
        name: str = None,
):
    """
    Similar to caching. Iterates over the whole dataset and creates a new
    dataset from the obtained examples. If the input dataset has `.items()`,
    the new dataset will be indexable and have `.keys()` and `.items()` (even
    if the input dataset did not have `.keys()`). If not, the new
    dataset will not have `.keys()`.

    Example:
        >>> ds = from_dataset(new([1, 2, 3, 4]))
        >>> ds
          ListDataset(len=4)
        MapDataset(_pickle.loads)
        >>> list(ds)
        [1, 2, 3, 4]
        >>> ds = from_dataset(new({'a': 1, 'b': 2, 'c': 3, 'd': 4}).map(lambda x: x**2))
        >>> ds
          DictDataset(len=4)
        MapDataset(_pickle.loads)
        >>> dict(ds)
        {'a': 1, 'b': 4, 'c': 9, 'd': 16}

        Works with filtered datasets:
        >>> ds = from_dataset(new([1, 2, 3, 4]).filter(lambda x: x%2))
        >>> list(ds)
        [1, 3]
        >>> ds = from_dataset(new({'a': 1, 'b': 2, 'c': 3, 'd': 4}).filter(lambda x: x%2))
        >>> dict(ds)
        {'a': 1, 'c': 3}
    """
    try:
        items = list(examples.items())
    except ItemsNotDefined:
        return from_list(list(examples),
                         immutable_warranty=immutable_warranty, name=name)
    else:
        return from_dict(dict(items),
                         immutable_warranty=immutable_warranty, name=name)


def concatenate(*datasets):
    """
    Create a new `Dataset` by concatenation of all passed datasets.

    Example:
        >>> import lazy_dataset
        >>> ds1 = lazy_dataset.new([1, 2, 3, 4])
        >>> ds2 = lazy_dataset.new([5, 6, 7, 8])
        >>> lazy_dataset.concatenate(ds1, ds2)
            ListDataset(len=4)
          MapDataset(_pickle.loads)
            ListDataset(len=4)
          MapDataset(_pickle.loads)
        ConcatenateDataset()

        >>> lazy_dataset.concatenate((ds1, ds2))
            ListDataset(len=4)
          MapDataset(_pickle.loads)
            ListDataset(len=4)
          MapDataset(_pickle.loads)
        ConcatenateDataset()

        >>> list(lazy_dataset.concatenate((ds1, ds2)))
        [1, 2, 3, 4, 5, 6, 7, 8]

    Args:
        datasets: List of datasets. Can be either a list of datasets
            (`concatenate((ds1, ds2, ...))`) or multiple datasets
            (`concatenate(ds1, ds2, ...)`)

    Returns:
        Concatenation of all input datasets

    """
    if len(datasets) == 0:
        raise ValueError('Need at least one dataset to concatenate!')
    if len(datasets) == 1 and isinstance(datasets[0], (tuple, list)):
        datasets, = datasets
    if not all([isinstance(dataset, Dataset) for dataset in datasets]):
        raise TypeError(
            f'All input arguments must be datasets! {Dataset} ' + ' '.join(
                str(type(d)) for d in datasets) + '|' + ' '.join(
                str(isinstance(d, Dataset)) for d in datasets))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatenateDataset(*datasets)


def intersperse(*datasets):
    """
    Intersperses datasets such that examples from each input dataset are
    evenly spaced in the output dataset.

    Args:
        *others: list of datasets to be interspersed

    Returns:
        `IntersperseDataset` combining examples of all provided datasets.

    Example:
        >>> import lazy_dataset
        >>> ds1 = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
        >>> ds2 = lazy_dataset.new({'f': 6, 'g': 7, 'h': 8})
        >>> interspersed = lazy_dataset.intersperse(ds1, ds2)
        >>> interspersed
            DictDataset(len=5)
          MapDataset(_pickle.loads)
            DictDataset(len=3)
          MapDataset(_pickle.loads)
        IntersperseDataset()
        >>> list(interspersed)
        [1, 6, 2, 3, 7, 4, 5, 8]
        >>> list(interspersed.keys())
        ['a', 'f', 'b', 'c', 'g', 'd', 'e', 'h']

    """
    if len(datasets) == 0:
        raise ValueError('Need at least one dataset to concatenate!')
    if len(datasets) == 1 and isinstance(datasets[0], (tuple, list)):
        datasets, = datasets
    if not all([isinstance(dataset, Dataset) for dataset in datasets]):
        raise TypeError(
            f'All input arguments must be datasets! {Dataset} ' + ' '.join(
                str(type(d)) for d in datasets) + '|' + ' '.join(
                str(isinstance(d, Dataset)) for d in datasets))
    if len(datasets) == 1:
        return datasets[0]
    return IntersperseDataset(*datasets)


def _zip(*datasets):
    """
    Create a new `Dataset` zipping all passed datasets.

    Example:
        >>> import lazy_dataset
        >>> ds1 = lazy_dataset.new({'1': 1, '2': 2, '3': 3, '4': 4})
        >>> ds2 = lazy_dataset.new({'1': 5, '2': 6, '3': 7, '4': 8})
        >>> lazy_dataset.zip(ds1, ds2)
            DictDataset(len=4)
          MapDataset(_pickle.loads)
            DictDataset(len=4)
          MapDataset(_pickle.loads)
        ZipDataset()

        >>> lazy_dataset.zip((ds1, ds2))
            DictDataset(len=4)
          MapDataset(_pickle.loads)
            DictDataset(len=4)
          MapDataset(_pickle.loads)
        ZipDataset()

        >>> list(lazy_dataset.zip(ds1, ds2))
        [(1, 5), (2, 6), (3, 7), (4, 8)]

    Args:
        datasets: List of datasets. Can be either a list of datasets
            (`zip_((ds1, ds2, ...))`) or multiple datasets
            (`zip_(ds1, ds2, ...)`)

    Returns:
        zip of all input datasets

    """
    if len(datasets) == 0:
        raise ValueError('Need at least one dataset to concatenate!')
    if len(datasets) == 1 and isinstance(datasets[0], (tuple, list)):
        datasets, = datasets
    if not all([isinstance(dataset, Dataset) for dataset in datasets]):
        raise TypeError(
            f'All input arguments must be datasets! {Dataset} ' + ' '.join(
                str(type(d)) for d in datasets) + '|' + ' '.join(
                str(isinstance(d, Dataset)) for d in datasets))
    return ZipDataset(*datasets)


def key_zip(*datasets):
    """
    Create a new `Dataset` zipping all passed datasets.

    Example:
        >>> import lazy_dataset
        >>> ds1 = lazy_dataset.new({'1': 1, '2': 2, '3': 3, '4': 4})
        >>> ds2 = lazy_dataset.new({'1': 5, '2': 6, '3': 7, '4': 8})
        >>> lazy_dataset.key_zip(ds1, ds2)
            DictDataset(len=4)
          MapDataset(_pickle.loads)
            DictDataset(len=4)
          MapDataset(_pickle.loads)
        KeyZipDataset()

        >>> lazy_dataset.key_zip((ds1, ds2))
            DictDataset(len=4)
          MapDataset(_pickle.loads)
            DictDataset(len=4)
          MapDataset(_pickle.loads)
        KeyZipDataset()

        >>> list(lazy_dataset.key_zip(ds1, ds2))
        [(1, 5), (2, 6), (3, 7), (4, 8)]

    Args:
        datasets: List of datasets. Can be either a list of datasets
            (`zip_((ds1, ds2, ...))`) or multiple datasets
            (`zip_(ds1, ds2, ...)`)

    Returns:
        zip of all input datasets

    """
    if len(datasets) == 0:
        raise ValueError('Need at least one dataset to concatenate!')
    if len(datasets) == 1 and isinstance(datasets[0], (tuple, list)):
        datasets, = datasets
    if not all([isinstance(dataset, Dataset) for dataset in datasets]):
        raise TypeError(
            f'All input arguments must be datasets! {Dataset} ' + ' '.join(
                str(type(d)) for d in datasets) + '|' + ' '.join(
                str(isinstance(d, Dataset)) for d in datasets))
    return KeyZipDataset(*datasets)


class FilterException(Exception):
    """
    Special Exception for the Dataset to indicate that this example should be
    skipped. The `Dataset.catch()` and
    `Dataset.prefetch(..., catch_filter_exception=True)` handle this
    exception.
    """
    pass


class Dataset:

    def copy(self, freeze: bool = False) -> 'Dataset':
        """
        Copies this dataset.

        Args:
            freeze: If `True`, the resulting dataset will not be random anymore.
                Only effects `ReShuffleDataset` at the moment.

        Returns:
            A copy of this dataset
        """
        raise NotImplementedError

    def __iter__(self, with_key=False):
        if with_key:
            # The ItemsDataset calls `__iter__(with_key=True)` to get the key
            # and the value of the dataset. The motivation is, that some
            # Datasets, e.g. FilterDataset, can yield key and value, while
            # keys is not known, because they depend von value.

            # Note: Consider the execution time, when implementing the
            # `__iter__` for a function, e.g. try to check only once, if
            # with_key is False and not inside the loop every time.

            # To be backward compatible, this function is called with:
            # `__iter__(self)` or `__iter_`_(self, with_key=True)`.
            # So Datasets can implement `__iter__` without `with_key`, when
            # it is not supported.

            # If a dataset does not support with_key==True, it should raise a
            # `_ItemsNotDefined(self.__class__.__name__)`. ItemDataset will
            # improve msg the exception.
            raise _ItemsNotDefined(self.__class__.__name__)
        raise NotImplementedError(
            f'__iter__ is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def __len__(self):
        # The correct exception type is TypeError and not NotImplementedError
        # for __len__. For example len(dataset) ignores TypeError but not
        # NotImplementedError
        raise TypeErrorLazyMessage(
            lambda: f'__len__ is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    @property
    def indexable(self) -> bool:
        raise NotImplementedError(
            f'indexable is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    @property
    def ordered(self) -> bool:
        """
        Indicate whether each iteration over the dataset yields the examples
        in the same order.

        This property is used for the `cache` implementation.

        Note: This property indicates only the ordered property from the
              dataset side.
              When a map function is not deterministic, i.e. has some random
              parts, this property should be false, but the dataset is unable
              to know this.
        """
        raise NotImplementedError(
            f'ordered is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def __getitem__(self, item):
        if isinstance(item, (slice, tuple, list)):
            return SliceDataset(item, self)
        elif isinstance(item, np.ndarray) and item.ndim == 1:
            return SliceDataset(item, self)
        elif isinstance(item, bytes):
            raise NotImplementedError(
                f'This is not implemented for an bytes objext. '
                f'Use bytes.decode() to convert it to an str.\n'
                f'__getitem__ is not implemented for {self.__class__}[{item!r}],\n'
                f'where type({item!r}) == {type(item)} '
                f'self: \n{self!r}'
            )
        raise NotImplementedError(
            f'__getitem__ is not implemented for {self.__class__}[{item!r}],\n'
            f'where type({item!r}) == {type(item)}\n'
            f'self:\n{self!r}'
        )

    def keys(self) -> list:
        raise NotImplementedError(
            f'keys is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def items(self) -> 'Dataset':
        """
        Returns:
             A `list` of key-value pairs (`tuple`s) like `dict.items()`.
             Only works for datasets that have `keys`.

        Example:
            >>> examples = {'a': {'d': 1}, 'b': {'e': 1}, 'c': {'f': 1}}
            >>> ds = DictDataset(examples)
            >>> list(ds)
            [{'d': 1}, {'e': 1}, {'f': 1}]
            >>> list(ds.items())
            [('a', {'d': 1}), ('b', {'e': 1}), ('c', {'f': 1})]
        """
        return ItemsDataset(self)

    def __contains__(self, item):
        # contains is not well defined for dataset, because dataset is a
        # mixture of tuple and dict. (tuple -> value, dict -> key)
        # Use the verbose contains (see exception msg)
        raise Exception(
            f"Use 'key in {self.__class__}.keys()' "
            f"instead of 'key in {self.__class__}'")

    def __call__(self):
        """

        Usecase:
          `tf.data.Dataset.from_generator(dataset)`

        Without __call__:
          `tf.data.Dataset.from_generator(lambda: dataset)`
        """
        return self.__iter__()

    def map(self, map_fn: callable, num_workers: int = 0,
            buffer_size: int = 100, backend: str = 't') -> 'Dataset':
        """
        Maps this dataset with `map_fn`. `map_fn` is applied to every element
        in the dataset and a new dataset is created with the results.

        Inspired by `map`.

        Args:
            map_fn: Function to transform an example dict. Takes a single
                example as provided by this dataset as its only positional
                argument and returns a transformed example, e.g., read and add
                 the observed audio signals.
            num_workers: If set to a value > 0, the `map_fn` is executed in
                parallel using `parallel_utils.lazy_parallel_map` with
                `num_workers` count of processes/threads.
            buffer_size: The size of the buffer used when `map_fn` is executed
                in parallel (`num_workers > 0`).
            backend: The backend used when `map_fn` is executed in parallel
                (`num_workers > 0`). See `parallel_utils.lazy_parallel_map` for
                details.

        Returns:
            MapDataset returning mapped examples. This can e.g. be used to read
            and add audio to the example dict (see read_audio method).

        Note:
          - `map_fn` can do inplace transformations without using copy.
            The `DictDataset` makes a deepcopy of each example and prevents a
            modification of the root example.
          - If `num_workers > 0` the `map_fn` is performed in parallel.
            But the input dataset is still executed serially.
            This allows an arbitrary input dataset. When it is desired to get
            an example in parallel, use prefetch on an indexable dataset.
          - `.map(map_fn).prefetch(...)` should be preferred over
            `.map(map_fn, num_workers=...)` where possible. The prefetch
            implementation executes all actions that were applied to the dataset
            before the prefetch operation (`map`, `batch`, ...) in parallel,
            but `.map(map_fn, num_workers=...)` only executes the passed
            `map_fn` in parallel and all functions previously applied to the
            dataset serially in the main thread.
        """
        if num_workers > 0:
            return ParMapDataset(
                map_fn, self, num_workers=num_workers, buffer_size=buffer_size,
                backend=backend
            )
        return MapDataset(map_fn, self)

    def batch_map(self, map_fn: callable, num_workers: int = 0,
                  buffer_size: int = 100, backend: str = 't') -> 'Dataset':
        """Applies map_fn to each element in each batch of the dataset.
        Requires dataset to be batched, i.e., the items of the
        dataset have to be iterable.

        Args:
            map_fn:
            num_workers:
            buffer_size:
            backend:

        Returns:

        >>> import lazy_dataset
        >>> ds = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3}, name='MyDataset')
        >>> ds
          DictDataset(name='MyDataset', len=3)
        MapDataset(_pickle.loads)
        >>> def foo(x): return 2*x
        >>> ds1 = ds.map(foo).batch(1)
        >>> ds1  # doctest: +ELLIPSIS
              DictDataset(name='MyDataset', len=3)
            MapDataset(_pickle.loads)
          MapDataset(<function foo at ...>)
        BatchDataset(batch_size=1)
        >>> list(ds1)
        [[2], [4], [6]]
        >>> ds2 = ds.batch(1).batch_map(foo)
        >>> ds2  # doctest: +ELLIPSIS
              DictDataset(name='MyDataset', len=3)
            MapDataset(_pickle.loads)
          BatchDataset(batch_size=1)
        MapDataset(_BatchMapWrapper(<function foo at ...>))
        >>> list(ds2)
        [[2], [4], [6]]
        """

        return self.map(
            _BatchMapWrapper(map_fn),
            num_workers=num_workers, buffer_size=buffer_size, backend=backend
        )

    def prefetch(self, num_workers: int, buffer_size: int, backend: str = 't',
                 catch_filter_exception: Any = None) -> 'Dataset':
        """
        Prefetches data (i.e., executes all actions applied previously with,
        e.g., `.map`, `.filter`, `.batch` or others) asynchronously in the
        background using `backend`.

        The dataset on which the `prefetch` method is used must be indexable
        (define `__getitem__`) and have a length (define `__len__`).
        When `num_workers == 1` and the `backend == 't'` a fallback
        implementation is used, that does not need these properties.

        For details on the available options for `backend` see
        `parallel_utils.lazy_parallel_map`.
        The threaded backend ('t') is recommended in most scenarios, especially
        if the mapped functions use a lot of I/O or numpy.

        Args:
            num_workers: Number of threads/processes used by the backend
            buffer_size: Number of elements that are prefetched and buffered
            backend: The used backend
            catch_filter_exception: If `True`, `FilterException`s are catched
                and the element that raised the exception while processing is
                dropped. This can also be set to a specific type (or a list of
                types) of exceptions to catch. If this is set to a value that
                evaluates to `True`, the resulting dataset does not have a
                length.

        Returns:
            Dataset that prefetches data in the background. This dataset is
            not indexable and only has a length if the input dataset defines
            a length and `bool(catch_filter_exception)` evaluates to `False`.

        Example:
            >>> import string
            >>> ascii = string.ascii_lowercase
            >>> ds = DictDataset({k: v for v, k in enumerate(ascii[:10])})
            >>> # ds1 = ds1.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> list(ds)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> def foo(ex):
            ...     print(f'called with {ex}')
            ...     return ex
            >>> ds = ds.map(foo)
            >>> list(ds)
            called with 0
            called with 1
            called with 2
            called with 3
            called with 4
            called with 5
            called with 6
            called with 7
            called with 8
            called with 9
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> ds = ds.prefetch(2, 4)
            >>> next(iter(ds))
            called with 0
            called with 1
            called with 2
            called with 3
            0

            # A second prefetch with multiple workes does not work, but a
            # single worker can use the iter protocol.
            >>> ds = DictDataset({k: v for v, k in enumerate(ascii[:10])})
            >>> ds = ds.prefetch(2, 4).prefetch(1, 4)
            >>> ds
                DictDataset(len=10)
              PrefetchDataset(2, 4, 't')
            PrefetchDataset(1, 4, 't')
            >>> next(iter(ds))
            0

        """
        return PrefetchDataset(
            input_dataset=self,
            num_workers=num_workers,
            buffer_size=buffer_size,
            backend=backend,
            catch_filter_exception=catch_filter_exception,
        )

    def filter(self, filter_fn: callable, lazy: bool = True) -> 'Dataset':
        """
        Filters elements in this dataset based on `filter_fn`. `filter_fn`
        must be a function that consumes (exactly) one example and returns
        a bool value. If the returned value is `True`, the example is kept,
        otherwise it is dropped.

        If using `lazy=False` this method executes all applied functions, so it
         should be called before applying expensive map functions.

        Syntax is inspired by:
        https://docs.python.org/3/library/functions.html#filter

        Args:
            filter_fn: Function to filter examples. Takes one example as input
                and returns `True` if example should be kept, and `False`
                otherwise
            lazy: If `True`, the computation is performed once the dataset
                visits the item and the resulting dataset does no longer have a
                length.

        Returns:
            `FilterDataset` iterating over filtered examples.

        Example:
            >>> import lazy_dataset
            >>> ds = lazy_dataset.new([1, 2, 3, 4, 5])
            >>> ds_filterd = ds.filter(lambda ex: ex != 2, lazy=False)
            >>> ds_filterd
                ListDataset(len=5)
              MapDataset(_pickle.loads)
            SliceDataset([0, 2, 3, 4])
            >>> list(ds_filterd)
            [1, 3, 4, 5]
            >>> ds_filterd = ds.filter(lambda ex: ex != 2, lazy=True)
            >>> ds_filterd  # doctest: +ELLIPSIS
                ListDataset(len=5)
              MapDataset(_pickle.loads)
            FilterDataset(<function <lambda> at 0x...>)
            >>> list(ds_filterd)
            [1, 3, 4, 5]

        """
        if lazy:
            # Input dataset can be indexable, but this is not needed.
            # Output still does not have `len` and is not indexable.
            return FilterDataset(filter_fn, self)
        else:
            # Input dataset needs to be indexable.
            if not self.indexable:
                raise RuntimeError(
                    'You can only use lazy=False if the incoming dataset is '
                    'indexable.'
                )
            idx = [i for i, e in enumerate(self) if filter_fn(e)]
            if len(self) > len(idx):
                LOG.info(f'Filtered {len(self) - len(idx)} of {len(self)} examples.')
            return self[idx]

    def catch(self, exceptions=FilterException,
              warn: bool = False) -> 'Dataset':
        """
        Drop examples that throw an exception (default: `FilterException`).
        This is an alternative to filter.

        Args:
            exceptions: One exception or a list of exceptions to filter
            warn: If `True`, enable logger warning when an exception is catched.

        Returns:
            A filtered dataset. The resulting dataset does no longer have a
            length because the resulting number of elements in the dataset
            cannot be determined beforehand.
        """
        return CatchExceptionDataset(self, exceptions=exceptions, warn=warn)

    def concatenate(self, *others) -> 'Dataset':
        """
        Concatenate this dataset with others. The keys of all datasets need to
        be unambiguous.

        Args:
            *others: list of datasets to be concatenated

        Returns:
            `ConcatenateDataset` that iterate over all examples of all provided
            datasets.

        Example:
            >>> import lazy_dataset
            >>> ds1 = lazy_dataset.new([1, 2, 3, 4, 5])
            >>> ds2 = lazy_dataset.new([6, 7, 8, 9, 0])
            >>> concatenated = ds1.concatenate(ds2)
            >>> concatenated
                ListDataset(len=5)
              MapDataset(_pickle.loads)
                ListDataset(len=5)
              MapDataset(_pickle.loads)
            ConcatenateDataset()
            >>> list(concatenated)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

        """
        if len(others) == 0:
            return self
        if len(others) == 1 and isinstance(others[0], (tuple, list)):
            others, = others
        return ConcatenateDataset(self, *others)

    def intersperse(self, *others) -> 'Dataset':
        """
        Intersperses datasets such that examples from each input dataset are
        evenly spaced in the output dataset.

        Args:
            *others: list of datasets to be interspersed

        Returns:
            `IntersperseDataset` combining examples of all provided datasets.

        Example:
            >>> import lazy_dataset
            >>> ds1 = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
            >>> ds2 = lazy_dataset.new({'f': 6, 'g': 7, 'h': 8})
            >>> interspersed = ds1.intersperse(ds2)
            >>> interspersed
                DictDataset(len=5)
              MapDataset(_pickle.loads)
                DictDataset(len=3)
              MapDataset(_pickle.loads)
            IntersperseDataset()
            >>> list(interspersed)
            [1, 6, 2, 3, 7, 4, 5, 8]
            >>> list(interspersed.keys())
            ['a', 'f', 'b', 'c', 'g', 'd', 'e', 'h']

        """
        if len(others) == 0:
            return self
        if len(others) == 1 and isinstance(others[0], (tuple, list)):
            others, = others
        return IntersperseDataset(self, *others)

    def zip(self, *others) -> 'Dataset':
        """
        Creates a `Dataset` by zipping together the given datasets.

        This method works similar to the python buildin zip except that it
        does not support a short zip, i.e. it asserts that all datasets have
        the same length.

        This function is usually followed by a map call to merge the tuple of
        dicts to a single dict.

        Considerations for the future:
             - Use Python 3.10 signature:
               https://www.python.org/dev/peps/pep-0618/
                   def zip(*iterables, strict=False):
                       ....
               But with the default `strict=True`.
        Args:
            *others: list of other datasets to be zipped

        Returns:
            ZipDataset

        Example:
            >>> ds1 = DictDataset({'a': {'z': 1}, 'b': {'z': 2}})
            >>> ds1 = ds1.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> ds2 = DictDataset({'a': {'y': 'c'}, 'b': {'y': 'd', 'z': 3}})
            >>> ds2 = ds2.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> ds3 = ds1.zip(ds2)
            >>> for e in ds3: print(e)
            ({'example_id': 'a', 'z': 1}, {'example_id': 'a', 'y': 'c'})
            ({'example_id': 'b', 'z': 2}, {'example_id': 'b', 'y': 'd', 'z': 3})

            # Merge the dicts, when conflict, prefer the second
            >>> ds4 = ds3.map(lambda example: {**example[0], **example[1]})
            >>> ds4  # doctest: +ELLIPSIS
                    DictDataset(len=2)
                  ItemsDataset()
                MapDataset(<function <lambda> at ...>)
                    DictDataset(len=2)
                  ItemsDataset()
                MapDataset(<function <lambda> at ...>)
              ZipDataset()
            MapDataset(<function <lambda> at ...>)
            >>> for e in ds4: print(e)
            {'example_id': 'a', 'z': 1, 'y': 'c'}
            {'example_id': 'b', 'z': 3, 'y': 'd'}

            # Lambda that merges an arbitary amount of dicts.
            >>> ds5 = ds3.map(lambda example: dict(sum([list(e.items()) for e in example], [])))
            >>> for e in ds5: print(e)
            {'example_id': 'a', 'z': 1, 'y': 'c'}
            {'example_id': 'b', 'z': 3, 'y': 'd'}
        """
        return ZipDataset(self, *others)

    def key_zip(self, *others) -> 'Dataset':
        """
        Creates a `Dataset` by zipping together the given datasets based on its
        keys.

        This method has two major differences to the built-in `zip()` function
        in Python. First, the zipping happens based on the keys of the
        first dataset (i.e. The first defines the order).
        Second, it assumes that all datasets have the same length and keys.

        This function is usually followed by a map call to merge the tuple of
        dicts to a single dict.

        Args:
            *others: list of other datasets to be zipped

        Returns:
            KeyZipDataset

        Example:
            >>> ds1 = DictDataset({'a': {'z': 1}, 'b': {'z': 2}})
            >>> ds1 = ds1.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> ds2 = DictDataset({'a': {'y': 'c'}, 'b': {'y': 'd', 'z': 3}})
            >>> ds2 = ds2.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> ds3 = ds1.key_zip(ds2)
            >>> for e in ds3: print(e)
            ({'example_id': 'a', 'z': 1}, {'example_id': 'a', 'y': 'c'})
            ({'example_id': 'b', 'z': 2}, {'example_id': 'b', 'y': 'd', 'z': 3})

            # Merge the dicts, when conflict, prefer the second
            >>> ds4 = ds3.map(lambda example: {**example[0], **example[1]})
            >>> ds4  # doctest: +ELLIPSIS
                    DictDataset(len=2)
                  ItemsDataset()
                MapDataset(<function <lambda> at ...>)
                    DictDataset(len=2)
                  ItemsDataset()
                MapDataset(<function <lambda> at ...>)
              KeyZipDataset()
            MapDataset(<function <lambda> at ...>)
            >>> for e in ds4: print(e)
            {'example_id': 'a', 'z': 1, 'y': 'c'}
            {'example_id': 'b', 'z': 3, 'y': 'd'}

            # Lambda that merges an arbitary amount of dicts.
            >>> ds5 = ds3.map(lambda example: dict(sum([list(e.items()) for e in example], [])))
            >>> for e in ds5: print(e)
            {'example_id': 'a', 'z': 1, 'y': 'c'}
            {'example_id': 'b', 'z': 3, 'y': 'd'}
        """
        return KeyZipDataset(self, *others)

    def shuffle(self, reshuffle: bool = False,
                rng: Optional[np.random.RandomState] = None,
                buffer_size: Optional[int] = None) -> 'Dataset':
        """
        Shuffle this dataset. This operation is not performed in-place, but
        returns a shuffled version of the original dataset.

        Args:
            reshuffle: If `True`, shuffle each time the dataset is iterated,
                but disable indexing. If `False`, single shuffle, but support
                indexing.
            rng: instance of `np.random.RandomState`.
                When `None`, fallback to np.random.
            buffer_size: If set, a local shuffle operation is used which only
                shuffles a window of size `buffer_size`. This is an
                approximation to the global shuffle.
                If set, `reshuffle` must be `True` and `rng` must be `None`.

        Returns:
            A `Dataset` with shuffled elements

        Note:
         - Use the `buffer_size` only in special cases where the dataset is
           already shuffled. For example a dataset is shuffled and then
           each example is split into multiple examples (using
           `.map(fragment_fn).unbatch()`). In this case a local shuffle
           (i.e., buffer_size > 0) is reasonable.

        Example:
            >>> np.random.seed(1)
            >>> examples = {'a': {}, 'b': {}, 'c': {}}
            >>> ds = DictDataset(examples)
            >>> ds = ds.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> ds = ds.shuffle(False)
            >>> ds  # doctest: +ELLIPSIS
                  DictDataset(len=3)
                ItemsDataset()
              MapDataset(<function <lambda> at ...>)
            SliceDataset([0 2 1])
            >>> list(ds)
            [{'example_id': 'a'}, {'example_id': 'c'}, {'example_id': 'b'}]
            >>> ds.keys()
            ('a', 'c', 'b')
        """
        # TODO: Should reshuffle default be True or False
        rng = np.random if rng is None else rng

        if buffer_size is not None:
            assert reshuffle is True, ('LocalShuffleDataset only supports '
                                       'reshuffle')
            return LocalShuffleDataset(self, buffer_size=buffer_size, rng=rng)

        if reshuffle is True:
            return ReShuffleDataset(self, rng=rng)
        elif reshuffle is False:
            permutation = np.arange(len(self))
            rng.shuffle(permutation)
            return self[permutation]
        else:
            raise ValueError(reshuffle, self)

    def tile(self, reps: int, shuffle: bool = False) -> 'Dataset':
        """
        Constructs an new dataset by repeating the dataset the number of
        times given by `reps`. This is done by copying the dataset and
        concatenating them.

        Args:
            reps: Number of repetitions
            shuffle: If `True`, calls shuffle with default arguments
                (*no reshuffle*) on each repetition prior to concatenation.

        Example:
            >>> import lazy_dataset
            >>> ds = lazy_dataset.new([1, 2, 3, 4, 5])
            >>> ds = ds.tile(reps=3)
            >>> ds
                ListDataset(len=5)
              MapDataset(_pickle.loads)
                ListDataset(len=5)
              MapDataset(_pickle.loads)
                ListDataset(len=5)
              MapDataset(_pickle.loads)
            ConcatenateDataset()
            >>> list(ds)
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        """
        datasets = [self] * reps
        if shuffle:
            datasets = [
                ds.shuffle()
                for ds in datasets
            ]
        return self.__class__.concatenate(*datasets)

    def groupby(self, group_fn: callable) -> Dict[Any, 'Dataset']:
        """
        Groups elements in the dataset using `group_fn`.

        `group_fn` takes exactly one example as its only positional argument and
        returns a group ID. This group ID can be any hashable value (i.e., any
        value that can be used as a key in a `dict`). All examples for
        which `group_fn` returns the same group ID are grouped into one
        `Dataset`.

        This method is inspired by `itertools.groupby`, where `group_fn`
        roughly behaves like the `key` function of `itertools.groupby`.

        Args:
            group_fn: A function which takes one element of the dataset and
                returns a hashable value as the group ID

        Returns:
            `dict` that maps from group ID to a `Dataset` that contains all
            elements that were mapped to the this group ID by `group_fn`

        Example:
            >>> examples = {'a': {'z': 1}, 'b': {'z': 2}, 'c': {'z': 1}, 'd': {'z': 1}, 'e': {'z': 3}}
            >>> ds = DictDataset(examples)
            >>> for k, v in ds.groupby(lambda ex: ex['z']).items():
            ...     print(f'{k}:', list(v), v.keys())
            ...     print(f'{v!r}')
            1: [{'z': 1}, {'z': 1}, {'z': 1}] ('a', 'c', 'd')
              DictDataset(len=5)
            SliceDataset([0, 2, 3])
            2: [{'z': 2}] ('b',)
              DictDataset(len=5)
            SliceDataset([1])
            3: [{'z': 3}] ('e',)
              DictDataset(len=5)
            SliceDataset([4])
        """
        iterable = enumerate(list(self.map(group_fn)))
        groups = collections.defaultdict(list)
        for k, g in itertools.groupby(iterable, lambda ele: ele[1]):
            indices = [ele[0] for ele in g]
            groups[k] += indices
        return {k: self[v] for k, v in groups.items()}

    def split(self, sections: int) -> List['Dataset']:
        """
        Splits the dataset into `sections` number of sections that have
        approximately equal length. The order of elements is not modified.

        Args:
            sections: Number of sections to divide this dataset into

        Returns:
            `list` of one `Dataset` for each section

        Example:
            >>> examples = {'a': {}, 'b': {}, 'c': {}, 'd': {}, 'e': {}}
            >>> ds = DictDataset(examples)
            >>> ds = ds.items().map(lambda x: {'example_id': x[0], **x[1]})
            >>> datasets = ds.split(2)
            >>> list(datasets[0])
            [{'example_id': 'a'}, {'example_id': 'b'}, {'example_id': 'c'}]
            >>> list(datasets[1])
            [{'example_id': 'd'}, {'example_id': 'e'}]
            >>> datasets[1].keys()
            ('d', 'e')
        """
        if sections < 1:
            raise ValueError("sections must be >= 1")
        if sections > len(self):
            raise ValueError(
                f'Dataset has only {len(self)} elements and cannot be '
                f'split into {sections} sections.'
            )
        slices = np.array_split(np.arange(len(self)), sections)
        return [self[s] for s in slices]

    def sort(self, key_fn: Optional[callable] = None,
             sort_fn: callable = sorted,
             reverse: bool = False) -> 'Dataset':
        """
        Sorts the dataset. The sort key is extracted from each example with
        the `key_fn`. The `sort_fn` allows to influence the sorting,
        e.g. `natsort.natsorted`.
        When the `key_fn` is `None`, the returned dataset is sorted according
        to `sort_fn(self.keys())`.

        Args:
            key_fn: Function that takes an element of this dataset and returns a
                key to sort by.
            sort_fn: Function used for sorting. Defaults to `sorted`, but can be
                set to, e.g., `natsort.natsorted`. The function must take a
                sequence of values and the keyword argument `reverse` and return
                a sorted sequence.
            reverse: If `True`, sort in reversed order.

        Returns:
            The sorted dataset

        Example:
            >>> examples = {'a': {'x': 1}, 'b': {'x': 3},  'c': {'x': 12}, 'd': {'x': 2}}
            >>> ds = DictDataset(examples)

            Sort by value
            >>> ds_sorted = ds.sort(lambda ex: ex['x'])
            >>> ds_sorted
              DictDataset(len=4)
            SliceDataset([0, 3, 1, 2])
            >>> print(ds_sorted.slice)
            [0 3 1 2]
            >>> dict(ds_sorted)
            {'a': {'x': 1}, 'd': {'x': 2}, 'b': {'x': 3}, 'c': {'x': 12}}

            Sort reversed by value
            >>> ds_sorted = ds.sort(lambda ex: ex['x'], reverse=True)
            >>> dict(ds_sorted)
            {'c': {'x': 12}, 'b': {'x': 3}, 'd': {'x': 2}, 'a': {'x': 1}}

            Sort by example key
            >>> dict(ds_sorted.sort())
            {'a': {'x': 1}, 'b': {'x': 3}, 'c': {'x': 12}, 'd': {'x': 2}}
        """
        if key_fn is None:
            try:
                keys = self.keys()
            except NotImplementedError as e:
                raise RuntimeError(
                    'dataset.sort(key_fn=None, ...) uses the keys\n'
                    'that belong to the examples to sort the dataset.\n'
                    'This dataset has no well defined keys.\n'
                    'Maybe you wanted to call\n'
                    '    `dataset.sort(key_fn=lambda example: ..., ...)`\n'
                    'to obtain the sort key from the example?\n'
                    f'self: \n{repr(self)}'
                ) from None
            sort_order = sort_fn(keys)
        else:
            sort_values = [key_fn(example) for example in self]
            sort_order = [
                index
                for _, index in sort_fn(
                    zip(sort_values, itertools.count()),
                    reverse=reverse,
                )
            ]
        return self[sort_order]

    def shard(self, num_shards, shard_index) -> 'Dataset':
        """
        Splits an dataset into `num_shards` shards and
        selects shard `shard_index`. Can be used to split the dataset
        between multiple processes (e.g. by using MPI). This is equivalent to
        `ds.split(num_shards)[shard_index]`.

        Args:
            num_shards: Number of shards
            shard_index: Shard index to select

        Returns:
            Shard number `shard_index`
        """
        return self.split(num_shards)[shard_index]

    def batch(self, batch_size: int, drop_last: bool = False) -> 'BatchDataset':
        """
        Create batches of size `batch_size` from the elements in this dataset.
        One batch is a list of elements of length `batch_size` (or slightly
        shorter for the last batch if `drop_last=False`). It usually makes sense
        to map a collate function after performing the batch operation.

        Args:
            batch_size: The size of the batches
            drop_last: If `True`, the last batch is dropped if it is smaller
                than `batch_size`

        Returns:
            Dataset of batches (lists of elements)

        Example:
            >>> examples = {'a': {'x': 1}, 'b': {'x': 3},  'c': {'x': 12}}
            >>> ds = DictDataset(examples)#.shuffle(reshuffle=True)
            >>> ds = ds.batch(2)
            >>> for ex in ds:
            ...     print(ex)
            [{'x': 1}, {'x': 3}]
            [{'x': 12}]
        """
        return BatchDataset(self, batch_size, drop_last)

    def batch_dynamic_bucket(
            self, bucket_cls, expiration=None, drop_incomplete=False,
            sort_key=None, reverse_sort=False, **bucket_kwargs) -> 'Dataset':
        """dynamically spawn and gather examples into buckets.
        
        Note that this operation is work in progress
        
        Args:
            bucket_cls: Bucket class to be used for bucketing. Must implement
                methods `maybe_append(example)` and `is_completed()`. The
                __init__ signature has to be `bucket_cls(init_example, **bucket_kwargs)`
                (e.g. `TimeSeriesBucket(init_example, batch_size, len_key, max_padding_rate, max_total_size=None)`)
            expiration: maximum life time of a bucket. After this number of
                subsequent examples it is either emitted
                (if drop_incomplete is False) or discarded
            drop_incomplete: if True drop incomplete buckets at the end of
                iteration or when buckets expire, else emit them.
            sort_key: optional callable or dict key returning a scalar to sort
                examples in bucket before emission.
            reverse_sort: if True and sort_key is not None, examples in bucket
                are sorted reversely before emission (e.g. pytorchs
                PackedSequence requires reversely sorted batches).

        Returns:
            A `Dataset` of batches (lists of elements)
        """
        return DynamicBucketDataset(
            self,
            bucket_cls=bucket_cls,
            expiration=expiration,
            drop_incomplete=drop_incomplete,
            sort_key=sort_key,
            reverse_sort=reverse_sort,
            **bucket_kwargs
        )

    def batch_dynamic_time_series_bucket(
            self, batch_size, len_key, max_padding_rate, max_total_size=None,
            expiration=None, drop_incomplete=False,
            sort_key=None, reverse_sort=False
    ) -> 'Dataset':
        """
        Wrapper for `batch_dynamic_bucket` using `DynamicTimeSeriesBucket`

        Args:
            batch_size: maximum number of examples in a batch (can be smaller
                if max_total_size is set)
            len_key: callable or dict key returning a scalar length given an
                example dict
            max_padding_rate: the maximum padding that has to be added to a
                signal in a bucket. E.g. if set to 0.2, an example of length
                100 can only be in a bucket with examples of lengths between
                80 and 125.
            max_total_size: maximum total size of a bucket
                (len(bucket)*max_length_in_bucket). If set, a bucket is
                completed if adding another example to the bucket would lead
                to exceeding max_total_size
            expiration: maximum life time of a bucket. After this number of
                subsequent examples it is either emitted
                (if drop_incomplete is False) or discarded
            drop_incomplete: if True drop incomplete buckets at the end of
                iteration or when buckets expire, else emit them.
            sort_key: optional callable or dict key returning a scalar to sort
                examples in bucket before emission.
            reverse_sort: if True and sort_key is not None, examples in bucket
                are sorted reversely before emission (e.g. pytorchs
                PackedSequence requires reversely sorted batches).

        Returns:
            A `Dataset` of batches (lists of elements)

        """
        return self.batch_dynamic_bucket(
            bucket_cls=DynamicTimeSeriesBucket, batch_size=batch_size,
            len_key=len_key, max_padding_rate=max_padding_rate,
            max_total_size=max_total_size,
            expiration=expiration, drop_incomplete=drop_incomplete,
            sort_key=sort_key, reverse_sort=reverse_sort
        )

    def unbatch(self) -> 'Dataset':
        """
        Divides a batch of examples into single examples, i.e. reverts
        `.batch()`.
        E.g., after splitting a (multi-channel) source example into a list of
        single channel examples using `.map(fragment_fn)`. The resulting
        `Dataset` does not implement `__len__`, because the count of resulting
        elements cannot be computed beforehand.

        Example:
            >>> examples = {'a': [1, 2], 'b': [3, 4]}
            >>> ds = DictDataset(examples)
            >>> list(ds)
            [[1, 2], [3, 4]]
            >>> list(ds.unbatch())
            [1, 2, 3, 4]
        """
        return UnbatchDataset(self)

    def __str__(self):
        return f'{self.__class__.__name__}()'

    def __repr__(self):
        # CB: Discussable, if this method name should be something like
        #     description instead of __repr__.
        import textwrap
        r = ''
        indent = '  '
        if hasattr(self, 'input_dataset'):
            s = repr(self.input_dataset)
            r += textwrap.indent(s, indent) + '\n'
        if hasattr(self, 'input_datasets'):
            for i, input_dataset in enumerate(self.input_datasets):
                if i >= 11:
                    r += textwrap.indent('...', indent) + '\n'
                    break
                s = repr(input_dataset)
                r += textwrap.indent(s, indent) + '\n'
        return r + str(self)

    def random_choice(
            self,
            size: Optional[int] = None,
            replace: bool = False,
            rng_state: np.random.RandomState = np.random,
    ):
        """
        Draws random samples from the dataset using the random number generator
        `rng_state`.

        Returns a random element of the dataset when size is None.
        When size is an integer, a random sub dataset is returned.
        Iterating two times over this dataset returns the same elements.


        Args:
            size: Size of the result. Must be smaller then `len(datset)` if
                `replace=False`.
            replace: Whether the examples are drawn with or without replacement.
                If `True`, an example can appear multiple times in the drawn
                output.
            rng_state: Used random number generator

        Returns:
            A `Dataset` that contains the drawn elements

        Example:
            >>> rng_state = np.random.RandomState(0)
            >>> examples = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
            >>> ds = DictDataset(examples)
            >>> def foo(ex):
            ...     print('foo')
            ...     return ex
            >>> ds = ds.map(foo)
            >>> print('Number', ds.random_choice(rng_state=rng_state))
            foo
            Number 3

            >>> print(ds.random_choice(1, rng_state=rng_state))
            SliceDataset([0])
            >>> print(ds.random_choice(2, rng_state=rng_state))
            SliceDataset([1 3])
            >>> ds_choice = ds.random_choice(7, rng_state=rng_state, replace=True)
            >>> print(ds_choice)
            SliceDataset([0 4 2 1 0 1 1])
            >>> print(list(ds_choice))
            foo
            foo
            foo
            foo
            foo
            foo
            foo
            [1, 5, 3, 2, 1, 2, 2]
        """
        i = rng_state.choice(len(self), size=size, replace=replace)
        return self[i]

    def apply(self, apply_fn: callable, lazy=False) -> 'Dataset':
        """
        Allows to apply functions to the complete dataset, not to the
        examples itself. Is equivalent to `dataset = apply_fn(dataset)`, but
        calls to `apply` can be easily chained. If `lazy` is True, then the
        `apply_fn` is applied before each iteration over the dataset.

        Args:
            apply_fn: For now, it is a single function, e.g.,
                `lambda ds: ds.shard(num_shards, shard_index)`   but can
                potentially be a list in future implementations.
            lazy: If True, the `apply_fn` is applied before each iteration.
                This allows to manually implement reshuffle (see
                doctext from `ApplyDataset`) or more advanced shuffle
                techniques.
        Returns:
            The transformed `Dataset`
        """
        if apply_fn is None:
            return self
        elif isinstance(apply_fn, list):
            raise NotImplementedError
        else:
            if lazy:
                return ApplyDataset(apply_fn, self)
            else:
                return apply_fn(self)

    def cache(
            self,
            lazy: bool = True,
            keep_mem_free: str = None,
    ) -> 'Dataset':
        """
        Caches data in memory. The dataset has to be indexable because the
        cache needs a unique identifier (key) for each example.

        It is recommended to apply caching after data loading and before any
        data-multiplying transformations (e.g., STFT) are applied, e.g.,
        `dataset.map(load_data).cache().map(transform)`.

        Warnings:
            This dataset is *not* immutable! It maintains the cache as an
            instance variable.

            This dataset freezes everything that comes before this dataset!
            E.g., anything random before applying `.cache` is frozen. Any
            shuffling has to be applied after caching!

            `keep_mem_free` has no effect, if `lazy=False`! If `lazy=False`,
            the dataset will always load all data.

        Examples:

            `Dataset.cache` can give a dataset that uses filter excpetions and
            thus has an unkown effective length the correct length again:
            >>> ds = new({'a': 1, 'b': 2, 'c': 3, 'd': 4})
            >>> def m(x):
            ...     if x % 2:
            ...         raise FilterException()
            ...     return x
            >>> ds = ds.map(m)
            >>> ds = ds.catch().cache(lazy=False)
            >>> len(ds)
            2

            Generate lots of data and hope that it doesn't crash
            >>> ds = new(list(range(1000)))
            >>> import numpy as np
            >>> ds = ds.map(lambda x: np.random.randn(1000, 1000))
            >>> ds = ds.cache(keep_mem_free='5 GB')
            >>> for example in ds:
            ...     pass # ...

        Args:
            lazy: If `True`, it caches "on the fly" and respects the setting
                of `keep_mem_free`. If `False`, it immediatly, on invocation
                of `cache`, loads all examples from the dataset into memory
                without respecting `keep_mem_free`.
            keep_mem_free: A human-friendly string containing a value and a
                unit ("<value><optional space><unit>"). Unit can be either
                "%" or any absolute byte unit (e.g., "B", "GB", "G"). If unit
                is "%", it keeps <value> percent of the memory free (e.g.,
                "50%"). If an absolute unit, it keeps that many bytes free
                (e.g., "5GB"). Defaults to "8 GB" if lazy=True.
        """
        if lazy:
            assert self.indexable, (
                'Lazy caching is only possible if the input dataset is '
                'indexable.'
            )
            return CacheDataset(self, keep_mem_free or "8 GB")
        else:
            assert not keep_mem_free, (
                'keep_mem_free is not supported for lazy=False'
            )
            assert self.indexable or self.ordered, (
                'Caching is only supported for indexable or ordered datasets.'
            )
            return new(self)

    def diskcache(
            self,
            cache_dir: Optional[Union[Path, str]] = None,
            reuse: bool = False, clear: bool = True
    ) -> 'Dataset':
        """
        Caches data in a local cache dir using the `diskcache` package. Only
        works with indexable datasets because caching requires a unique key for
        each element.

        It is recommended to apply caching after data loading and before any
        data-multiplying transformations (e.g., STFT) are applied, e.g.,
        `dataset.map(load_data).diskcache().map(transform)` to minimize the
        cache size.

        Examples:
            >>> ds = new(list(range(10))).diskcache()

        Warnings:
            Be careful to only enalbe `resue` when you know that the data in
            `cache_dir` is valid for your dataset! Otherwise this option
            produces invalid outputs and hard-to-find bugs!

            This dataset is *not* immutable! It maintains the cache as an
            instance variable, possibly even across runs of your script (if
            `reuse=True`)!

            This dataset freezes everything that comes before this dataset!
            E.g., anything random before applying `.diskcache` is frozen. Any
            shuffling has to be applied after caching!

        Args:
            cache_dir: Directory to save the cached data. If `None`, it uses
                "/tmp/<somerandomid>".
            reuse: If `True`, data found in `cache_dir` is re-used. If `False`,
                it raises an exception when `cache_dir` exists.
            clear: If `True`, it tries to clear the cache directory on exit.
                This works for the usual exit methods (i.e., normal exit,
                keyboard interrupt) but not reliably for other signals (e.g.,
                SIGTERM, SIGKILL, SIGSEGV). Clearing on SIGTERM usually works
                when the signal is handled somewhere in the python code and
                fails otherwise (Adding
                `signal.signal(signal.SIGTERM, lambda *x: exit(1))` makes the
                the clearing process work but might interfere with other parts
                of the program than try to handle signals).
        """
        return DiskCacheDataset(self, cache_dir, reuse, clear)


class KeyErrorCloseMatches(KeyError):
    # Improve the Exception msg for KeyErrors
    def __str__(self):
        if len(self.args) == 2 and isinstance(self.args[0], str):
            item, keys = self.args
            import difflib
            # Suggestions are sorted by their similarity.
            try:
                suggestions = difflib.get_close_matches(
                    item, keys, cutoff=0, n=100
                )
            except TypeError:
                keys = map(str, keys)
                suggestions = difflib.get_close_matches(
                    item, keys, cutoff=0, n=100
                )
            return f'Invalid key {item!r}.\n' \
                   f'Close matches: {suggestions!r}.'
        else:
            return super().__str__()


class TypeErrorLazyMessage(TypeError):
    def __init__(self, message_creator: callable):
        self.message_creator = message_creator

    def __str__(self):
        return self.message_creator()


class DictDataset(Dataset):
    """
    Dataset to iterate over a dict of examples dicts.
    """

    def __init__(self, examples, name=None):
        assert isinstance(examples, dict), (type(examples), examples)
        self.examples = examples
        self.name = name
        self._keys = tuple(self.examples.keys())

    def copy(self, freeze=False):
        # Use __new__ to avoid a copy of keys
        new = self.__class__.__new__(self.__class__)
        new.examples = self.examples
        new.name = self.name
        new._keys = self.keys()
        return new

    @property
    def indexable(self):
        return True

    @property
    def ordered(self):
        return True

    def __str__(self):
        if self.name is None:
            return f'{self.__class__.__name__}(len={len(self)})'
        else:
            return f'{self.__class__.__name__}' \
                   f'(name={self.name!r}, len={len(self)})'

    def keys(self):
        return self._keys

    def __iter__(self, with_key=False):
        if with_key:
            for k in self.keys():
                yield k, self[k]
        else:
            for k in self.keys():
                yield self[k]

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                example = self.examples[item]
            except KeyError:
                raise KeyErrorCloseMatches(item, self.keys()) from None
        elif isinstance(item, numbers.Integral):
            key = self.keys()[item]
            example = self.examples[key]
        else:
            return super().__getitem__(item)

        # Assumes that the example is immutable.
        # See from_dict(immutable_warranty).
        return example

    def __len__(self):
        return len(self.examples)


class ListDataset(Dataset):
    """
    Dataset to iterate over a list of examples with each example being a dict
    according to the json structure as outline in the top of this file.
    """

    def __init__(self, examples, name=None):
        assert isinstance(examples, (tuple, list)), (type(examples), examples)
        self.examples = examples
        self.name = name

    def copy(self, freeze=False):
        return self.__class__(self.examples, name=self.name)

    @property
    def indexable(self):
        return True

    @property
    def ordered(self) -> bool:
        return True

    def __str__(self):
        if self.name is None:
            return f'{self.__class__.__name__}(len={len(self)})'
        else:
            return f'{self.__class__.__name__}' \
                   f'(name={self.name}, len={len(self)})'

    def __iter__(self, with_key=False):
        if with_key:
            raise _ItemsNotDefined(self.__class__.__name__)

        yield from self.examples

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            example = self.examples[item]
        else:
            return super().__getitem__(item)

        # Assumes that the example is immutable.
        # See from_dict(immutable_warranty).
        return example

    def __len__(self):
        return len(self.examples)


class MapDataset(Dataset):
    """
    Dataset that iterates over an input_dataset and applies a transformation
    map_function to each element.

    """

    def __init__(self, map_function, input_dataset):
        """

        Args:
            map_function: function that transforms an element of input_dataset.
                Use deepcopy within the map_function if necessary.
            input_dataset: any dataset (e.g. DictDataset)

        """
        assert callable(map_function), map_function
        self.map_function = map_function
        self.input_dataset = input_dataset

    def copy(self, freeze=False):
        return self.__class__(
            self.map_function,
            input_dataset=self.input_dataset.copy(freeze=freeze)
        )

    @property
    def indexable(self):
        return self.input_dataset.indexable

    @property
    def ordered(self) -> bool:
        # This is only true if the mapped function is deterministic!
        return self.input_dataset.ordered

    def __str__(self):
        map_function_str = str(self.map_function)
        if (
                'built-in function' in map_function_str
                or self.map_function == deepcopy
        ):
            map_function_str = (
                f'{self.map_function.__module__}'
                f'.{self.map_function.__qualname__}'
            )
        return f'{self.__class__.__name__}({map_function_str})'

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self, with_key=False):
        if with_key:
            for k, v in self.input_dataset.__iter__(with_key=True):
                yield k, self.map_function(v)
        else:
            yield from map(self.map_function, self.input_dataset)

    def keys(self):
        return self.input_dataset.keys()

    def __getitem__(self, item):
        if isinstance(item, (str, numbers.Integral)):
            return self.map_function(self.input_dataset[item])
        else:
            return super().__getitem__(item)


class ParMapDataset(MapDataset):
    """
    Should this dataset support getitem? Getitem disables the buffer.

    >>> ds = new({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6})
    >>> def foo(ex): return ex
    >>> ds = ds.map(foo, num_workers=2, buffer_size=4)
    >>> list(ds)
    [1, 2, 3, 4, 5, 6]
    >>> list(ds.items())
    [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6)]
    """

    def __init__(
            self, map_function, input_dataset, num_workers, buffer_size,
            backend='t'
    ):
        super().__init__(map_function, input_dataset)
        assert num_workers >= 1
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.backend = backend

    def copy(self, freeze=False):
        return self.__class__(
            self.map_function,
            input_dataset=self.input_dataset.copy(freeze=freeze),
            num_workers=self.num_workers,
            buffer_size=self.buffer_size,
            backend=self.backend,
        )

    @staticmethod
    def _with_key_map_function(key_ex, func):
        # Avoid a lambda function, because it makes problems with
        # multiprocessing.
        key, ex = key_ex
        ex = func(ex)
        return key, ex

    def __iter__(self, with_key=False):
        from lazy_dataset.parallel_utils import lazy_parallel_map
        if with_key:
            return lazy_parallel_map(
                functools.partial(
                    self._with_key_map_function, func=self.map_function),
                self.input_dataset.__iter__(with_key=True),
                buffer_size=self.buffer_size,
                max_workers=self.num_workers,
                backend=self.backend,
            )
        else:
            return lazy_parallel_map(
                self.map_function,
                self.input_dataset,
                buffer_size=self.buffer_size,
                max_workers=self.num_workers,
                backend=self.backend,
            )


class _BatchMapWrapper:
    def __init__(self, map_fn):
        self.map_fn = map_fn

    def __call__(self, batch):
        return [self.map_fn(ex) for ex in batch]

    def __str__(self):
        return f'{self.__class__.__name__}({self.map_fn})'


class ApplyDataset(Dataset):
    """
    Dataset that applies a function to the dataset, before iterating over it.

    Motivation: Each iteration should be different, like in reshuffle, but
                with more control by the user.

    Example, how apply can be used to implement reshuffle:

        >>> ds = new({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6})
        >>> class ReShuffle:
        ...     def __init__(self, rng):
        ...         self.rng = rng
        ...         self.permutation = None
        ...     def __call__(self, ds):
        ...         if self.permutation is None:
        ...             self.permutation = np.arange(len(ds))
        ...         self.rng.shuffle(self.permutation)
        ...         return ds[self.permutation]
        >>> rng = np.random.RandomState(0)
        >>> ds = ds.apply(ReShuffle(rng), lazy=True)
        >>> ds  # doctest: +ELLIPSIS
            DictDataset(len=6)
          MapDataset(_pickle.loads)
        ApplyDataset(<...core.ReShuffle object at 0x...>)
        >>> print(list(ds))
        [6, 3, 2, 4, 1, 5]
        >>> print(list(ds))
        [3, 4, 1, 6, 2, 5]
        >>> print(list(ds))
        [6, 5, 4, 1, 2, 3]

        # And now the same example with `ds.shuffle(True, rng)`.
        >>> ds = new({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6})
        >>> rng = np.random.RandomState(0)
        >>> ds = ds.shuffle(True, rng)
        >>> print(list(ds))
        [6, 3, 2, 4, 1, 5]
        >>> print(list(ds))
        [3, 4, 1, 6, 2, 5]
        >>> print(list(ds))
        [6, 5, 4, 1, 2, 3]
    """

    def __init__(self, apply_function, input_dataset):
        """

        Args:
            apply_function: function that transforms the input_dataset.
            input_dataset: any dataset (e.g. DictDataset)

        """
        assert callable(apply_function), apply_function
        self.apply_function = apply_function
        self.input_dataset = input_dataset

    def copy(self, freeze=False):
        if freeze:
            return self.apply_function(self.input_dataset).copy(freeze=freeze)
        else:
            return self.__class__(
                self.apply_function,
                input_dataset=self.input_dataset.copy(freeze=freeze)
            )

    @property
    def indexable(self) -> bool:
        # Unknown, but most likely False, otherwise a non lazy apply would be
        # used.
        return False

    @property
    def ordered(self) -> bool:
        # Unknown, but most likely False, otherwise a non lazy apply would be
        # used.
        return False

    def __str__(self):
        apply_function_str = str(self.apply_function)
        if 'built-in function' in apply_function_str:
            apply_function_str = (
                f'{self.apply_function.__module__}'
                f'.{self.apply_function.__qualname__}'
            )
        return f'{self.__class__.__name__}({apply_function_str})'

    # apply_function may change __len__
    # def __len__(self):
    #      return len(self.input_dataset)

    def __iter__(self, with_key=False):
        frozen = self.copy(freeze=True)
        for example in frozen.__iter__(with_key=with_key):
            yield example

    # apply_function may change keys
    # def keys(self):
    #     return self.input_dataset.keys()

    # def __getitem__(self, item):
    #     return super().__getitem__(item)


class CatchExceptionDataset(Dataset):
    """
    >>> from lazy_dataset.core import DictDataset, FilterException
    >>> ds = DictDataset({'a': 1, 'b': 2, 'c': 3})
    >>> list(ds)
    [1, 2, 3]
    >>> def foo(integer):
    ...     if integer == 2:
    ...         raise FilterException('Exception msg')
    ...     else:
    ...         return integer
    >>> list(ds.map(foo))
    Traceback (most recent call last):
    ...
    lazy_dataset.core.FilterException: Exception msg
    >>> list(ds.map(foo).catch())
    [1, 3]
    >>> ds.map(foo).catch()[0]  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    NotImplementedError: __getitem__ is not well defined for <class 'lazy_dataset.core.CatchExceptionDataset'>[0],
    because 0 is an index
    self:
        DictDataset(len=3)
      MapDataset(<function foo at ...>)
    CatchExceptionDataset()
    """

    def __init__(
            self,
            input_dataset,
            exceptions=FilterException,
            warn=False
    ):
        self.input_dataset = input_dataset
        self.exceptions = exceptions
        self.warn = warn

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            exceptions=self.exceptions,
            warn=self.warn,
        )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def __getitem__(self, item):
        if isinstance(item, (str)):
            return self.input_dataset[item]
        elif isinstance(item, numbers.Integral):
            raise NotImplementedError(
                f'__getitem__ is not well defined for '
                f'{self.__class__}[{item!r}],\n'
                f'because {item!r} is an index\n'
                f'self:\n{self!r}'
            )
        else:
            return super().__getitem__(item)

    def __iter__(self, with_key=False):
        input_dataset = self.input_dataset.copy(freeze=True)
        catched_count = 0
        total_count = 0
        if with_key:
            for k in input_dataset.keys():
                total_count += 1
                try:
                    yield k, input_dataset[k]
                except self.exceptions as e:
                    catched_count += 1
                    if self.warn:
                        msg = repr(e)
                        LOG.warning(msg)
        else:
            for i in range(len(input_dataset)):
                total_count += 1
                try:
                    yield input_dataset[i]
                except self.exceptions as e:
                    catched_count += 1
                    if self.warn:
                        msg = repr(e)
                        LOG.warning(msg)
        if catched_count > 0:
            types = ', '.join([exception.__name__ for exception in self.exceptions]) \
                if isinstance(self.exceptions, (list, tuple)) \
                else self.exceptions.__name__
            LOG.info(f'{self.__class__.__name__} filtered {catched_count} of {total_count} examples (catched expections: {types}).')


class PrefetchDataset(Dataset):
    def __init__(
            self,
            input_dataset,
            num_workers,
            buffer_size,
            backend='t',
            catch_filter_exception=False,
    ):
        if num_workers == 1 and backend == 't':
            pass
        else:
            # Input dataset needs to be indexable.
            try:
                _ = len(input_dataset)
            except Exception:
                # Note: Indexable is not the necessary property, but I don't know
                #       how to shortly describe that the frozen dataset must be
                #       indexable.
                raise RuntimeError(
                    'You can only use Prefetch with multiple workers if the'
                    'incoming dataset is indexable.\n'
                    f'input_dataset:\n{input_dataset!r}'
                )
        assert num_workers >= 1, num_workers
        assert buffer_size >= num_workers, (num_workers, buffer_size)

        self.input_dataset = input_dataset
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.backend = backend
        self.catch_filter_exception = catch_filter_exception

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            num_workers=self.num_workers,
            buffer_size=self.buffer_size,
            backend=self.backend,
            catch_filter_exception=self.catch_filter_exception,
        )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def __len__(self):
        if self.catch_filter_exception:
            raise TypeErrorLazyMessage(
                lambda: f'__len__ is not implemented for {self.__class__} ' +
                f'if `catch_filter_exception` is set.\n' +
                f'self: \n{repr(self)}'
            )
        else:
            return len(self.input_dataset)

    def __iter__(self, with_key=False):
        if self.num_workers == 1 and self.backend == 't':
            yield from self._single_thread_prefetch()
            return

        # Convert ReShuffleDataset to ShuffleDataset
        input_dataset = self.input_dataset.copy(freeze=True)

        from lazy_dataset.parallel_utils import lazy_parallel_map

        if with_key:
            iterable = self.keys()
        else:
            iterable = range(len(self.input_dataset))

        if not self.catch_filter_exception:
            if with_key:
                def function(key):
                    return key, input_dataset[key]
            else:
                function = input_dataset.__getitem__

            yield from lazy_parallel_map(
                function,
                iterable,
                buffer_size=self.buffer_size,
                max_workers=self.num_workers,
                backend=self.backend,
            )
        else:
            if self.catch_filter_exception is True:
                catch_filter_exception = FilterException
            else:
                catch_filter_exception = self.catch_filter_exception

            unique_object = object()

            if with_key:
                def catcher(key):
                    try:
                        return key, input_dataset[key]
                    except catch_filter_exception:
                        return unique_object
            else:
                def catcher(index):
                    try:
                        return input_dataset[index]
                    except catch_filter_exception:
                        return unique_object

            catched_count = 0
            total_count = 0
            for data in lazy_parallel_map(
                catcher,
                iterable,
                buffer_size=self.buffer_size,
                max_workers=self.num_workers,
                backend=self.backend,
            ):
                total_count += 1
                if data is unique_object:
                    catched_count += 1
                else:
                    yield data
            if catched_count > 0:
                types = ', '.join([exception.__name__ for exception in catch_filter_exception]) \
                    if isinstance(catch_filter_exception, (list, tuple)) \
                    else catch_filter_exception.__name__
                LOG.info(f'{self.__class__.__name__} filtered {catched_count} of {total_count} examples (catched exceptions: {types}).')

    def _single_thread_prefetch(self):
        """
                >>> import string
        >>> ascii = string.ascii_lowercase
        >>> ds = DictDataset({k: v for v, k in enumerate(ascii[:5])})
        >>> list(ds)
        [0, 1, 2, 3, 4]
        >>> def foo(ex):
        ...     print(f'called with {ex}')
        ...     return ex
        >>> ds = ds.map(foo)
        >>> ds = PrefetchDataset(ds.filter(lambda x: x != 2), 1, 2)
        >>> next(iter(ds))  # The buffer is filled with [1, 3]. 2 is ignored.
        called with 0
        called with 1
        called with 2
        called with 3
        0
        >>> list(ds)  # Note: 2 is missing
        called with 0
        called with 1
        called with 2
        called with 3
        called with 4
        [0, 1, 3, 4]
        """
        from lazy_dataset.parallel_utils import single_thread_prefetch

        if self.catch_filter_exception:
            if self.catch_filter_exception is True:
                exceptions = FilterException
            else:
                exceptions = self.catch_filter_exception
            input_dataset = CatchExceptionDataset(
                self.input_dataset,
                exceptions=exceptions,
            )
        else:
            input_dataset = self.input_dataset

        return single_thread_prefetch(input_dataset, self.buffer_size)

    def __str__(self):
        return (
            f'{self.__class__.__name__}'
            f'({self.num_workers}, {self.buffer_size}, {self.backend!r})'
        )


class ReShuffleDataset(Dataset):
    """
    Dataset that shuffles the input_dataset. Assumes, that the input_dataset
    has a length.
    Note:
        This Dataset reshuffle each iteration, but does not support indexing.
    """

    def __init__(self, input_dataset, rng=np.random):
        self._permutation = np.arange(len(input_dataset))
        self.input_dataset = input_dataset
        self.rng = rng

    @property
    def permutation(self):
        self.rng.shuffle(self._permutation)
        return self._permutation

    def copy(self, freeze=False):
        if freeze:
            return self.input_dataset.copy(freeze=freeze)[self.permutation]
        else:
            return self.__class__(
                input_dataset=self.input_dataset.copy(freeze=freeze),
            )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        return False

    def __len__(self):
        return len(self.input_dataset)

    # keys is not well defined for this dataset
    # The First dataset (i.e. DictDataset has sorted keys), so what should
    # this dataset return? Maybe a frozenset to highlight unordered?
    # def keys(self):
    #     return frozenset(self.input_dataset.keys())

    def __iter__(self, with_key=False):
        if with_key:
            keys = self.input_dataset.keys()
            for idx in self.permutation:
                k = keys[idx]
                yield k, self.input_dataset[idx]
        else:
            for idx in self.permutation:
                yield self.input_dataset[idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_dataset[item]
        elif isinstance(item, numbers.Integral):
            raise TypeError(
                f'{self.__class__.__name__} does not support '
                f'integers as argument of __getitem__.'
                f'Got argument "{item}" of type {type(item)}.'
            )
        else:
            # Let super().__getitem__(...) raise the Exception when item is a
            # slice, tuple or list.
            return super().__getitem__(item)


class LocalShuffleDataset(Dataset):
    """
    Dataset that shuffles the input_dataset locally by randomly sampling from
    a fixed length buffer. Hence also applicable to Datasets that does not
    support indexing
    Note:
        This Dataset reshuffles each iteration, but does not support indexing.


    >>> ds_list = new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> np.random.seed(0)
    >>> ds = ds_list.shuffle(reshuffle=True, buffer_size=3)
    >>> ds
        ListDataset(len=13)
      MapDataset(_pickle.loads)
    LocalShuffleDataset(buffer_size=3)
    >>> list(ds)
    [0, 2, 1, 4, 5, 7, 3, 9, 6, 8, 10, 12, 11]
    >>> list(ds)
    [1, 3, 4, 0, 5, 6, 7, 8, 2, 10, 9, 11, 12]
    >>> ds = ds_list.shuffle(
    ...     reshuffle=True, buffer_size=3, rng=np.random.RandomState(0))
    >>> ds
        ListDataset(len=13)
      MapDataset(_pickle.loads)
    LocalShuffleDataset(buffer_size=3, rng=RandomState(MT19937))
    >>> list(ds)
    [0, 2, 1, 4, 5, 7, 3, 9, 6, 8, 10, 12, 11]
    >>> list(ds)
    [1, 3, 4, 0, 5, 6, 7, 8, 2, 10, 9, 11, 12]

    """

    def __init__(self, input_dataset, buffer_size=100, rng=np.random):
        self.input_dataset = input_dataset
        self.buffer_size = buffer_size
        self.rng = rng

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            buffer_size=self.buffer_size,
        )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        return False

    def __str__(self):
        if self.rng == np.random:
            sig = f'buffer_size={self.buffer_size}'
        else:
            sig = f'buffer_size={self.buffer_size}, rng={self.rng}'

        return f'{self.__class__.__name__}({sig})'

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self, with_key=False):
        buffer = list()
        if with_key:
            iterator = self.input_dataset.__iter__(with_key=True)
        else:
            iterator = iter(self.input_dataset)

        for element in iterator:
            buffer.append(element)
            if len(buffer) >= self.buffer_size:
                yield buffer.pop(int(self.rng.choice(self.buffer_size)))
        self.rng.shuffle(buffer)
        for element in buffer:
            yield element

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_dataset[item]
        elif isinstance(item, numbers.Integral):
            raise TypeError(
                f'{self.__class__.__name__} does not support '
                f'integers as argument of __getitem__.'
                f'Got argument "{item}" of type {type(item)}.'
            )
        else:
            # Let super().__getitem__(...) raise the Exception when item is a
            # slice, tuple or list.
            return super().__getitem__(item)


class SliceDataset(Dataset):
    def __init__(self, slice, input_dataset: Dataset):
        """
        Should not be used directly. Simply call the dataset with brackets:
        dataset[0:10:2]
        dataset[slice(0, None, 2)]  # Uncommon

        It allows any kind of Numpy style indexing:
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.indexing.html

        Args:
            slice: Can be a slice, e.g. `slice(0, None, 2)`.
            input_dataset:
        """
        if not input_dataset.indexable:
            raise RuntimeError(
                f'You tried `dataset[{slice}]`\n'
                'You can only use `__getitem__` on datasets that are '
                'indexable.\n'
                'Example datasets that do not support indexing are:\n'
                f'  {FilterDataset.__name__}, {ReShuffleDataset.__name__} and '
                f'{PrefetchDataset.__name__}\n'
                'The following dataset is not indexable:\n'
                f'{input_dataset!r}'
            )

        self._slice = slice
        if np.ndim(self._slice) == 2:
            assert len(self._slice) == 1, self._slice
            self._slice, = self._slice

        try:
            self.slice = np.arange(len(input_dataset))[self._slice,]
        except IndexError:
            if isinstance(slice, (tuple, list)) and isinstance(slice[0], str):
                # Assume sequence of str
                keys = {k: i for i, k in enumerate(input_dataset.keys())}
                self.slice = operator.itemgetter(*slice)(keys)
                if len(slice) == 1:
                    self.slice = (self.slice,)
            else:
                raise

        self.input_dataset = input_dataset

    def copy(self, freeze=False):
        # Use __new__ to avoid a copy of slice (and __init__ overhead)
        new = self.__class__.__new__(self.__class__)
        new.input_dataset = self.input_dataset.copy(freeze=freeze)
        new._slice = self._slice
        new.slice = self.slice
        return new

    @property
    def indexable(self):
        assert self.input_dataset.indexable, (
            self.input_dataset.indexable, self.input_dataset)
        return True

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    _keys = None

    def keys(self):
        if self._keys is None:
            keys = self.input_dataset.keys()
            # itemgetter makes the same as
            # "tuple([keys[i] for i in self.slice])"
            # but is 10 times faster
            self._keys = operator.itemgetter(*self.slice)(keys)
            if len(self.slice) == 1:
                self._keys = (self._keys,)
        return self._keys

    def __len__(self):
        return len(self.slice)

    def __str__(self):
        if isinstance(self._slice, (tuple, list)):
            slice_str = textwrap.shorten(
                str(self._slice[:20])[1:-1],
                width=50,
                placeholder=' ...',
            )
            slice_str = f'[{slice_str}]'
        else:
            slice_str = str(self._slice)

        return f'{self.__class__.__name__}({slice_str})'

    def __iter__(self, with_key=False):
        if with_key:
            keys = self.input_dataset.keys()
            for idx in self.slice:
                yield keys[idx], self.input_dataset[idx]
        else:
            for idx in self.slice:
                yield self.input_dataset[idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_dataset[item]
        elif isinstance(item, numbers.Integral):
            return self.input_dataset[self.slice[item]]
        else:
            return super().__getitem__(item)


class FilterDataset(Dataset):
    """
    Dataset that iterates only over those elements of input_dataset that meet
    filter_function.
    """

    def __init__(self, filter_function, input_dataset):
        """

        Args:
            filter_function: a function that takes an element of the input
                dataset and returns True if the element is valid else False.
            input_dataset: any dataset (e.g. DictDataset)

        """
        assert callable(filter_function), filter_function
        self.filter_function = filter_function
        self.input_dataset = input_dataset

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            filter_function=self.filter_function,
        )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def __str__(self):
        return f'{self.__class__.__name__}({self.filter_function})'

    def __iter__(self, with_key=False):
        filtered_count = 0
        total_count = 0
        if with_key:
            for key, example in self.input_dataset.__iter__(with_key=True):
                total_count += 1
                if self.filter_function(example):
                    yield key, example
                else:
                    filtered_count += 1
        else:
            for example in self.input_dataset:
                total_count += 1
                if self.filter_function(example):
                    yield example
                else:
                    filtered_count += 1
        if filtered_count > 0:
            LOG.info(f'{self.__class__.__name__} filtered {filtered_count} of {total_count} examples.')

    def __getitem__(self, item):
        assert isinstance(item, str), (
            f'key == {item!r}\n{self.__class__} does not support __getitem__ '
            f'for type(key) == {type(item)},\n'
            f'Only type str is allowed.\n'
            f'self:\n{repr(self)}'
        )
        ex = self.input_dataset[item]
        if not self.filter_function(ex):
            raise IndexError(item)
        return ex


class ConcatenateDataset(Dataset):
    """
    Iterates over all elements of all input_datasets.
    Best use is to concatenate cross validation or evaluation datasets.
    It does not work well with buffer based shuffle (i.e. in Tensorflow).

    Here, __getitem__ for str is not possible per definition when IDs collide.
    """

    def __init__(self, *input_datasets):
        """

        Args:
            *input_datasets: list of datasets

        """
        self.input_datasets = input_datasets

    def copy(self, freeze=False):
        return self.__class__(
            *[ds.copy(freeze=freeze) for ds in self.input_datasets]
        )

    @property
    def indexable(self):
        return all([ds.indexable for ds in self.input_datasets])

    @property
    def ordered(self) -> bool:
        return all(ds.ordered for ds in self.input_datasets)

    def __iter__(self, with_key=False):
        for input_dataset in self.input_datasets:
            if with_key:
                iterable = input_dataset.__iter__(with_key=True)
            else:
                iterable = input_dataset
            for example in iterable:
                yield example

    def __len__(self):
        return sum([len(i) for i in self.input_datasets])

    _keys = None

    def keys(self):
        """
        >>> examples = {'a': 1, 'b': 2, 'c': 3}
        >>> ds = DictDataset(examples)
        >>> ds.concatenate(ds).keys()
        Traceback (most recent call last):
        ...
        AssertionError: Keys are not unique. There are 3 duplicates.
        ['a', 'b', 'c']
        >>> list(ds.concatenate(ds.map(lambda x: x+10)))
        [1, 2, 3, 11, 12, 13]
        """
        if self._keys is None:
            keys = []
            for dataset in self.input_datasets:
                keys += list(dataset.keys())
            if len(keys) != len(set(keys)):
                duplicates = [
                    item  # https://stackoverflow.com/a/9835819/5766934
                    for item, count in collections.Counter(keys).items()
                    if count > 1
                ]
                duplicates_str = textwrap.shorten(
                    str(duplicates)[1:-1], width=500, placeholder=' ...')
                raise AssertionError(
                    f'Keys are not unique. '
                    f'There are {len(duplicates)} duplicates.'
                    f'\n[{duplicates_str}]'
                )
            self._keys = tuple(keys)
        return self._keys

    def __getitem__(self, item):
        """
        >>> ds1 = DictDataset({'a': {}, 'b': {}})
        >>> ds1 = ds1.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> ds2 = DictDataset({'c': {}, 'd': {}})
        >>> ds2 = ds2.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> ds = ds1.concatenate(ds2)
        >>> ds['a']
        {'example_id': 'a'}
        >>> ds['c']
        {'example_id': 'c'}
        >>> ds[5]
        Traceback (most recent call last):
          ...
        IndexError: 5
        >>> ds[-1]
        {'example_id': 'd'}
        >>> ds[-5]
        Traceback (most recent call last):
          ...
        IndexError: -5

        """
        if isinstance(item, numbers.Integral):
            _item = item
            if item < 0:
                item = item + len(self)
                if item < 0:
                    # Without this check, you could still get a value if
                    # item < -len(self), but always from the first
                    # dataset which is wrong.
                    raise IndexError(_item)
            for dataset in self.input_datasets:
                if len(dataset) <= item:
                    item -= len(dataset)
                else:
                    return dataset[item]
            raise IndexError(_item)
        elif isinstance(item, str):
            self.keys()  # test unique keys
            for dataset in self.input_datasets:
                if item in dataset.keys():
                    return dataset[item]
            # In collections.ChainMap is
            # 'try: ... except KeyError: ...'
            # used, since a dataset should provide a better exception msg,
            # __contains__ is faster than collections.ChainMap
            # because the overhead of calculating the exception msg is too high.

            if item in self.keys():
                raise Exception(
                    f'There is an internal error in {self.__class__}. '
                    f'Could not find {item} in input datasets, but it is in '
                    f'{self.keys()}'
                )
            raise KeyErrorCloseMatches(item, self.keys())
        else:
            return super().__getitem__(item)


class IntersperseDataset(Dataset):
    """
    See Dataset.intersperse

    >>> examples = {'a': 1, 'b': 2, 'c': 3}
    >>> ds = DictDataset(examples)
    >>> IntersperseDataset(ds, ds).keys()
    Traceback (most recent call last):
    ...
    AssertionError: Keys are not unique. There are 3 duplicates.
    ['a', 'b', 'c']
    >>> list(IntersperseDataset(ds, ds.map(lambda x: x+10)))
    [1, 11, 2, 12, 3, 13]
    >>> list(IntersperseDataset(IntersperseDataset(ds, ds), ds.map(lambda x: x+10)))
    [1, 1, 11, 2, 2, 12, 3, 3, 13]
    """

    def __init__(self, *input_datasets):
        """

        Args:
            *input_datasets: list of datasets

        """
        self.input_datasets = input_datasets
        assert len(self.input_datasets) >= 1, (f'You have to provide at least '
                                               f'one dataset.'
                                               f'\n{self.input_datasets}')
        assert all([len(dataset) > 0 for dataset in self.input_datasets])
        dataset_lengths = [len(ds) for ds in input_datasets]
        self.order = sorted([
            ((example_index + 1) / ds_len, dataset_index, example_index)
            for dataset_index, ds_len in enumerate(dataset_lengths)
            for example_index in range(ds_len)
        ])

    def copy(self, freeze=False):
        # Use __new__ to avoid a copy of order (and __init__ overhead)
        new = self.__class__.__new__(self.__class__)
        new.order = self.order
        new.input_datasets = [ds.copy(freeze=freeze) for ds in self.input_datasets]
        return new

    _keys = None

    def keys(self):
        if self._keys is None:
            ds_keys = [list(ds.keys()) for ds in self.input_datasets]
            keys = [
                ds_keys[dataset_idx][example_idx]
                for _, dataset_idx, example_idx in self.order
            ]
            if len(keys) != len(set(keys)):
                duplicates = [
                    item  # https://stackoverflow.com/a/9835819/5766934
                    for item, count in collections.Counter(keys).items()
                    if count > 1
                ]
                duplicates_str = textwrap.shorten(
                    str(duplicates)[1:-1], width=500, placeholder=' ...')
                raise AssertionError(
                    f'Keys are not unique. '
                    f'There are {len(duplicates)} duplicates.'
                    f'\n[{duplicates_str}]'
                )
            self._keys = tuple(keys)
        return self._keys

    @property
    def indexable(self):
        return all([dataset.indexable for dataset in self.input_datasets])

    @property
    def ordered(self) -> bool:
        return all(ds.ordered for ds in self.input_datasets)

    def __iter__(self, with_key=False):
        if with_key:
            iterators = [
                ds.__iter__(with_key=True) for ds in self.input_datasets]
        else:
            iterators = [iter(ds) for ds in self.input_datasets]
        for _, dataset_idx, _ in self.order:
            yield next(iterators[dataset_idx])

    def __len__(self):
        return len(self.order)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            _, dataset_idx, example_idx = self.order[item]
            return self.input_datasets[dataset_idx][example_idx]
        elif isinstance(item, str):
            self.keys()  # test unique keys
            for dataset in self.input_datasets:
                if item in dataset.keys():
                    return dataset[item]
        else:
            return super().__getitem__(item)


class ZipDataset(Dataset):
    """
    See Dataset.zip
    """

    def __init__(self, *input_datasets):
        """

        Args:
            *input_datasets: list of datasets

        """
        self.input_datasets = input_datasets
        assert len(self.input_datasets) >= 1, (f'You have to provide at least '
                                               f'one dataset.'
                                               f'\n{self.input_datasets}')
        lengths = [len(ds) for ds in input_datasets]
        assert len(set(lengths)) == 1, lengths

    def copy(self, freeze=False):
        return self.__class__(
            *[ds.copy(freeze=freeze) for ds in self.input_datasets]
        )

    @property
    def indexable(self):
        return all([dataset.indexable for dataset in self.input_datasets])

    @property
    def ordered(self) -> bool:
        return all(ds.ordered for ds in self.input_datasets)

    def __iter__(self, with_key=False):
        if with_key:
            raise _ItemsNotDefined(self.__class__.__name__)
        for examples in zip(*self.input_datasets):
            yield examples

    def __len__(self):
        return len(self.input_datasets[0])

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return tuple([
                ds[item] for ds in self.input_datasets
            ])
        else:
            return super().__getitem__(item)


class KeyZipDataset(Dataset):
    """
    While ZipDataset combines examples from multiple datasets index wise this
    Datasets combines examples key wise, i.e. input_datasets are expected to
    have the same keys.
    """

    def __init__(self, *input_datasets):
        """

        Args:
            *input_datasets: list of datasets

        """
        self.input_datasets = input_datasets
        assert len(self.input_datasets) >= 1, (f'You have to provide at least '
                                               f'one dataset.'
                                               f'\n{self.input_datasets}')
        assert len(self.input_datasets) >= 2, (f'Currently limited to at least '
                                               f'two dataset. Could be removed.'
                                               f'\n{self.input_datasets}')
        keys = set.union(*[set(ds.keys()) for ds in self.input_datasets])
        lengths = [
            len(keys - set(ds.keys())) for ds in self.input_datasets
        ]
        if set(lengths) != {0}:
            missing_keys = [
                keys - set(ds.keys()) for ds in self.input_datasets
            ]
            raise AssertionError(
                f'Expect that all input_datasets have the same keys. '
                f'Missing keys: '
                f'{missing_keys}\n{self.input_datasets}'
            )

    def copy(self, freeze=False):
        return self.__class__(
            *[ds.copy(freeze=freeze) for ds in self.input_datasets]
        )

    @property
    def indexable(self):
        return all([ds.indexable for ds in self.input_datasets])

    @property
    def ordered(self) -> bool:
        return True

    def __iter__(self, with_key=False):
        if with_key:
            for key in self.keys():
                yield key, tuple([
                    ds[key]
                    for ds in self.input_datasets
                ])
        else:
            for key in self.keys():
                yield tuple([
                    ds[key]
                    for ds in self.input_datasets
                ])

    def __len__(self):
        return len(self.input_datasets[0])

    _keys = None

    def keys(self):
        if self._keys is None:
            self._keys = self.input_datasets[0].keys()
        return self._keys

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            item = self.keys()[item]
        if isinstance(item, str):
            return tuple([
                ds[item]
                for ds in self.input_datasets
            ])
        else:
            return super().__getitem__(item)


class ItemsNotDefined(Exception):
    """
    Special Exception for the Dataset to indicate that a dataset does not
    support items.
    """
    pass


class _ItemsNotDefined(BaseException):
    """
    Special Exception for the Dataset to indicate that a dataset does not
    support items.

    This is the internal exception, that shouldn't be caught by the user or
    the `dataset.catch(Exception)`, hence base class is BaseException.
    """
    pass


class ItemsDataset(Dataset):
    """
    >>> ds_plain = new({'a': 1, 'b': 2, 'c': 3})
    >>> ds = ds_plain.filter(lambda x: True).items()
    >>> ds  # doctest: +ELLIPSIS
          DictDataset(len=3)
        MapDataset(_pickle.loads)
      FilterDataset(<function <lambda> at 0x...>)
    ItemsDataset()
    >>> list(ds)
    [('a', 1), ('b', 2), ('c', 3)]

    >>> ds = ds_plain.shuffle(True, rng=np.random.RandomState(0)).items()
    >>> list(ds)
    [('c', 3), ('b', 2), ('a', 1)]
    >>> list(ds)
    [('a', 1), ('c', 3), ('b', 2)]

    >>> ds_nokeys = ds_plain.filter(lambda x: True)  # No keys and no len
    >>> ds_nokeys_rng = ds_plain.shuffle(True, rng=np.random.RandomState(0))  # No keys
    >>> list(ds_nokeys.map(lambda x: x + 10).items())
    [('a', 11), ('b', 12), ('c', 13)]
    >>> list(ds_nokeys.concatenate(ds_plain).items())
    [('a', 1), ('b', 2), ('c', 3), ('a', 1), ('b', 2), ('c', 3)]
    >>> list(ds_nokeys_rng.intersperse(ds_nokeys_rng).items())
    [('c', 3), ('a', 1), ('c', 3), ('c', 3), ('b', 2), ('b', 2)]
    >>> list(ds_plain.key_zip(ds_plain).items())
    [('a', (1, 1)), ('b', (2, 2)), ('c', (3, 3))]
    >>> list(ds_nokeys_rng.catch().items())
    [('a', 1), ('b', 2), ('c', 3)]
    >>> list(ds_plain[:2].items())
    [('a', 1), ('b', 2)]

    >>> ds_plain[:2].items()[0]
    ('a', 1)
    >>> ds_plain[:2].items()['a']
    ('a', 1)

    """
    def __init__(self, input_dataset):
        self.input_dataset = input_dataset
        # ToDo: assert input_dataset supports .items(), at the moment we have
        #       no property that indicate this.

    def copy(self, freeze=False):
        return self.__class__(self.input_dataset.copy(freeze=freeze))

    @property
    def indexable(self):
        return self.input_dataset.indexable

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def keys(self):
        return self.input_dataset.keys()

    def __getitem__(self, item):
        if isinstance(item, str):
            return item, self.input_dataset[self.keys().index(item)]
        elif isinstance(item, numbers.Integral):
            return self.keys()[item], self.input_dataset[item]
        else:
            return super().__getitem__(item)

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self, with_key=False):
        """
        >>> ds = new({'a': 1, 'b': 2, 'c': 3})
        >>> list(ds.items())
        [('a', 1), ('b', 2), ('c', 3)]
        >>> list(ds.items().items())
        [('a', ('a', 1)), ('b', ('b', 2)), ('c', ('c', 3))]
        """
        if with_key:
            for k, v in self:
                yield k, (k, v)
        else:
            try:
                yield from self.input_dataset.__iter__(with_key=True)
            except _ItemsNotDefined as e:
                # `[].items()` raises AttributeError, but here we raise the
                # error later.
                raise ItemsNotDefined(
                    f'`.items()` can only be used, when each input dataset\n'
                    f"supports `.items()`. At least one doesn't support it.\n"
                    f'self: \n{repr(self)}'
                ) from e


class BatchDataset(Dataset):
    """

    >>> from lazy_dataset.core import DictDataset
    >>> import string
    >>> examples = {c: i for i, c in enumerate(string.ascii_letters[:7])}
    >>> ds = DictDataset(examples)
    >>> ds = ds.batch(3)
    >>> ds
      DictDataset(len=7)
    BatchDataset(batch_size=3)
    >>> list(ds), len(ds)
    ([[0, 1, 2], [3, 4, 5], [6]], 3)
    >>> ds[2], ds[-1]
    ([6], [6])
    >>> ds[3]
    Traceback (most recent call last):
    ...
    IndexError: tuple index out of range
    >>> ds = DictDataset(examples)
    >>> ds = ds.batch(3, drop_last=True)
    >>> list(ds), len(ds)
    ([[0, 1, 2], [3, 4, 5]], 2)
    >>> ds[-1]
    [3, 4, 5]
    >>> ds = DictDataset(examples)[:6]
    >>> ds = ds.batch(3)
    >>> list(ds), len(ds)
    ([[0, 1, 2], [3, 4, 5]], 2)
    >>> ds[1]
    [3, 4, 5]
    >>> ds['abc']
    Traceback (most recent call last):
    ...
    NotImplementedError: __getitem__ is not implemented for <class 'lazy_dataset.core.BatchDataset'>['abc'],
    where type('abc') == <class 'str'>
    self:
        DictDataset(len=7)
      SliceDataset(slice(None, 6, None))
    BatchDataset(batch_size=3)
    >>> ds[-3]
    Traceback (most recent call last):
      ...
    IndexError: -3
    """

    def __init__(self, input_dataset, batch_size, drop_last=False):
        self.input_dataset = input_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    @property
    def indexable(self):
        return self.input_dataset.input_dataset

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def __str__(self):
        return f'{self.__class__.__name__}(batch_size={self.batch_size})'

    def __iter__(self, with_key=False):
        if with_key:
            raise _ItemsNotDefined(self.__class__.__name__)
        current_batch = list()
        for element in self.input_dataset:
            current_batch.append(element)
            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = list()
        if len(current_batch) > 0 and not self.drop_last:
            yield current_batch

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            if item < 0:
                # only touch len when necessary
                item = item + len(self)
                if item < 0:
                    raise IndexError(item - len(self))
            input_index = item * self.batch_size
            current_batch = []
            for i in range(self.batch_size):
                try:
                    current_batch.append(self.input_dataset[input_index + i])
                except IndexError:
                    if i == 0 or self.drop_last:
                        raise
                    else:
                        pass
            return current_batch
        # elif isinstance(index, str):
        # ToDo: allow merge/collate keys -> allows __getitem__(str)
        else:
            return super().__getitem__(item)

    def __len__(self):
        length = len(self.input_dataset) / self.batch_size
        if self.drop_last:
            return int(length)
        return int(np.ceil(length))


class UnbatchDataset(Dataset):
    """
    Divides a batch of examples into single examples.

    >>> ds = ListDataset([
    ...     (1, 2),  # tuple
    ...     [4, 5],  # list
    ...     zip([6, 7, 8]),  # zip
    ...     range(8, 10),  # range
    ...     (i for i in range(10, 12)),  # collections.abc.Generator
    ... ]).unbatch()
    >>> ds
      ListDataset(len=5)
    UnbatchDataset()
    >>> list(ds)
    [1, 2, 4, 5, (6,), (7,), (8,), 8, 9, 10, 11]

    """

    def __init__(self, input_dataset):
        self.input_dataset = input_dataset

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze)
        )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def __iter__(self, with_key=False):
        if with_key:
            raise _ItemsNotDefined(self.__class__.__name__)

        for batch in self.input_dataset:
            # Don't support `dict` and `str`.
            # While
            # assert not isinstance(batch, (dict, str)), (type(batch), batch)
            # would work, use for now a whitelist instead of a blacklist:
            assert isinstance(batch, (
                list,
                tuple,
                collections.abc.Generator,
                zip,
                range,
            )), (type(batch), batch)
            for example in batch:
                yield example


class DynamicBucket:
    def __init__(self, init_example, batch_size):
        """
        Base Bucket to inherit from when defining custom buckets.

        Args:
            init_example: first example in the batch
            batch_size: number of examples in a batch
        """
        self.data = [init_example]
        self.batch_size = batch_size

    def is_completed(self):
        return len(self.data) >= self.batch_size

    def assess(self, example):
        raise NotImplementedError

    def _append(self, example):
        self.data.append(example)

    def maybe_append(self, example):
        assert not self.is_completed()
        if self.assess(example):
            self._append(example)
            return True
        return False


class DynamicTimeSeriesBucket(DynamicBucket):
    def __init__(
            self, init_example, batch_size, len_key, max_padding_rate,
            max_total_size=None
    ):
        """
        Bucket of examples with similar sequence lengths to prevent excessive
        padding.

        Args:
            init_example: first example in the batch
            batch_size: maximum number of examples in a batch (can be smaller
                if max_total_size is set)
            len_key: callable or dict key returning a scalar length given an
                example dict
            max_padding_rate: the maximum padding that has to be added to a
                signal in a bucket. E.g. if set to 0.2, an example of length
                100 can only be in a bucket with examples of lengths between
                80 and 125.
            max_total_size: maximum total size of a bucket
                (len(bucket)*max_length_in_bucket). If set, a bucket is
                completed if adding another example to the bucket would lead
                to exceeding max_total_size

        """
        super().__init__(init_example, batch_size)
        self.len_key = len_key if callable(len_key) else (lambda x: x[len_key])
        self.max_padding_rate = max_padding_rate
        self.max_total_size = max_total_size

        init_len = self.len_key(init_example)
        self.lower_bound = init_len * (1 - self.max_padding_rate)
        self.upper_bound = init_len / (1 - self.max_padding_rate)
        self.max_len = init_len

    def is_completed(self):
        return (
            super().is_completed()
            or (
                self.max_total_size is not None
                and ((len(self.data) + 1) * self.max_len > self.max_total_size)
            )
        )

    def assess(self, example):
        seq_len = self.len_key(example)
        return self.lower_bound <= seq_len <= self.upper_bound

    def _append(self, example):
        super()._append(example)
        seq_len = self.len_key(example)
        self.lower_bound = max(
            self.lower_bound, seq_len * (1 - self.max_padding_rate)
        )
        self.upper_bound = min(
            self.upper_bound, seq_len / (1 - self.max_padding_rate)
        )
        self.max_len = max(self.max_len, seq_len)


class DynamicBucketDataset(Dataset):
    """
    >>> examples = [1, 10, 5, 7, 8, 2, 4]
    >>> batch_dataset = DynamicBucketDataset(\
        examples, DynamicTimeSeriesBucket, batch_size=2, len_key=lambda x: x, max_padding_rate=0.5)
    >>> [batch for batch in batch_dataset]
    [[10, 5], [7, 8], [1, 2], [4]]
    >>> batch_dataset = DynamicBucketDataset(\
    examples, DynamicTimeSeriesBucket, batch_size=2, len_key=lambda x: x, max_padding_rate=0.5, drop_incomplete=True)
    >>> [batch for batch in batch_dataset]
    [[10, 5], [7, 8], [1, 2]]
    >>> batch_dataset = DynamicBucketDataset(\
    examples, DynamicTimeSeriesBucket, batch_size=2, len_key=lambda x: x, max_padding_rate=0.2)
    >>> [batch for batch in batch_dataset]
    [[10, 8], [5, 4], [1], [7], [2]]
    >>> batch_dataset = DynamicBucketDataset(\
    examples, DynamicTimeSeriesBucket, expiration=4, batch_size=2, len_key=lambda x: x, max_padding_rate=0.2)
    >>> [batch for batch in batch_dataset]
    [[10, 8], [1], [5, 4], [7], [2]]
    """

    def __init__(
            self, input_dataset, bucket_cls,
            expiration=None, drop_incomplete=False,
            sort_key=None, reverse_sort=False, **bucket_kwargs
    ):
        """dynamically spawn and gather examples into buckets.
        Note that this class is work in progress
        Args:
            input_dataset:
            bucket_cls: Bucket class to be used for bucketing. Must implement
                methods `maybe_append(example)` and `is_completed()`. The
                __init__ signature has to be `bucket_cls(init_example, **bucket_kwargs)`
                (e.g. `TimeSeriesBucket(init_example, batch_size, len_key, max_padding_rate, max_total_size=None)`)
            expiration: maximum life time of a bucket. After this number of
                subsequent examples it is either emitted
                (if drop_incomplete is False) or discarded
            drop_incomplete: if True drop incomplete buckets at the end of
                iteration or when buckets expire, else emit them.
            sort_key: optional callable or dict key returning a scalar to sort
                examples in bucket before emission.
            reverse_sort: if True and sort_key is not None, examples in bucket
                are sorted reversely before emission (e.g. pytorchs
                PackedSequence requires reversely sorted batches).
        """
        self.input_dataset = input_dataset
        self.expiration = expiration
        self.drop_incomplete = drop_incomplete
        self.sort_key = sort_key if (callable(sort_key) or sort_key is None) \
            else (lambda x: x[sort_key])
        self.reverse_sort = reverse_sort
        self.bucket_cls = bucket_cls
        self.bucket_kwargs = bucket_kwargs

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            bucket_cls=self.bucket_cls,
            expiration=self.expiration,
            drop_incomplete=self.drop_incomplete,
            sort_key=self.sort_key,
            reverse_sort=self.reverse_sort,
            **self.bucket_kwargs
        )

    @property
    def indexable(self):
        return False

    @property
    def ordered(self) -> bool:
        # This is only true if the bucket is deterministic!
        return self.input_dataset.ordered

    def __iter__(self, with_key=False):
        if with_key:
            raise _ItemsNotDefined(self.__class__.__name__)
        buckets = list()
        dropped_count = 0
        total_count = 0
        for i, example in enumerate(self.input_dataset):
            total_count += 1
            bucket = None
            for j, (bucket_j, _) in enumerate(buckets):
                if bucket_j.maybe_append(example):
                    bucket = bucket_j
                    break
            if bucket is None:
                bucket = self.bucket_cls(example, **self.bucket_kwargs)
                buckets.append((bucket, i))
                j = len(buckets) - 1

            if bucket.is_completed():
                data = bucket.data
                if self.sort_key is not None:
                    data = sorted(data, key=self.sort_key, reverse=self.reverse_sort)
                yield data
                buckets.pop(j)

            if self.expiration is not None:
                for j, (bucket, creation_idx) in enumerate(buckets):
                    if (i - creation_idx) >= self.expiration:
                        data = bucket.data
                        if not self.drop_incomplete:
                            if self.sort_key is not None:
                                data = sorted(data, key=self.sort_key, reverse=self.reverse_sort)
                            yield data
                        else:
                            dropped_count += len(data)
                        buckets.pop(j)
                        break

        for bucket, _ in buckets:
            data = bucket.data
            if not self.drop_incomplete:
                if self.sort_key is not None:
                    data = sorted(data, key=self.sort_key, reverse=self.reverse_sort)
                yield data
            else:
                dropped_count += len(data)
        if dropped_count > 0:
            LOG.info(f'{self.__class__.__name__} dropped {dropped_count} of {total_count} examples.')


class _CacheWrapper:
    def __init__(self, immutable_warranty: str = 'pickle'):
        self._serialize, self._deserialize = _get_serialize_and_deserialize(
            immutable_warranty)
        self.cache = {}

    def __getitem__(self, item):
        return self._deserialize(self.cache[item])

    def __setitem__(self, key, value):
        self.cache[key] = self._serialize(value)

    def __contains__(self, item):
        return item in self.cache

    def __len__(self):
        return len(self.cache)


class CacheDataset(Dataset):
    def __init__(self, input_dataset: Dataset, keep_mem_free=None,
                 immutable_warranty: str = 'pickle') -> None:
        assert input_dataset.indexable, (
            'CacheDataset only works if dataset is indexable!'
        )
        self.input_dataset = input_dataset
        self._cache = _CacheWrapper(immutable_warranty)
        self._keep_mem_free = self._get_memory_size(keep_mem_free)

    @property
    def indexable(self) -> bool:
        return self.input_dataset.indexable

    @property
    def ordered(self) -> bool:
        return self.input_dataset.ordered

    def keys(self) -> list:
        return self.input_dataset.keys()

    @staticmethod
    def _get_memory_size(keep_mem_free):
        if keep_mem_free is None:
            return None

        if isinstance(keep_mem_free, int):
            return keep_mem_free
        elif keep_mem_free.strip().endswith('%'):
            import psutil
            value = float(keep_mem_free.strip(' %'))
            assert 0 <= value <= 100, value
            return psutil.virtual_memory().total * value / 100
        else:
            import humanfriendly
            return humanfriendly.parse_size(keep_mem_free, binary=True)

    _do_cache = True

    def check(self):
        if self._keep_mem_free is None:
            return True

        if not self._do_cache:
            return False

        import psutil
        if psutil.virtual_memory().available <= self._keep_mem_free:
            # Return without writing to cache if there is not enough
            # free memory
            import warnings
            warnings.warn(
                'Max capacity of the in-memory cache is reached. '
                'The remaining data will not be cached.',
                ResourceWarning
            )
            self._do_cache = False
            return False
        return True

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.keys().index(item)

        if isinstance(item, numbers.Integral):
            try:
                return self._cache[item]
            except KeyError:
                value = self.input_dataset[item]
                if self.check():
                    self._cache[item] = value
                return value
        else:
            # Support for slices etc.
            return super().__getitem__(item)

    def __iter__(self, with_key=False):
        if with_key:
            keys = self.keys()
            for i in range(len(self)):
                yield keys[i], self[i]
        else:
            for i in range(len(self)):
                yield self[i]

    def __len__(self):
        return len(self.input_dataset)

    def copy(self, freeze: bool = False) -> 'Dataset':
        if not freeze:
            import warnings
            warnings.warn(
                'Copying a CacheDataset preserves the cache, i.e., the '
                'already cached part of the dataset will be frozen even if '
                'freeze=False!'
            )
        # We have to share the cache here because otherwise a new cache would
        # be initialized at every copy and copy is called by prefetch before
        # iterating over the dataset
        copy = self.__class__.__new__(self.__class__)
        copy.input_dataset = self.input_dataset.copy(freeze)
        copy._cache = self._cache
        copy._keep_mem_free = self._keep_mem_free
        return copy

    def __str__(self):
        import humanfriendly
        if self._keep_mem_free:
            return (
                f'{self.__class__.__name__}(keep_free='
                f'{humanfriendly.format_size(self._keep_mem_free, binary=True)}'
                f')'
            )
        else:
            return super().__str__()


class _DiskCacheWrapper:
    """
    Wraps a `diskcache.Cache` and takes care of cleaning up when destroyed.
    The wrapper is required to support sharing the cache among copies of the
    dataset.
    """
    def __init__(self, cache_dir, reuse, clear):
        self.clear = clear
        self.reuse = reuse

        import diskcache
        if cache_dir is not None and Path(cache_dir).is_dir() and len(
                list(Path(cache_dir).glob('*'))) > 0:
            if reuse:
                LOG.info(f'Cache dir "{cache_dir}" already exists. Re-using stored data.')
            else:
                raise RuntimeError(
                    f'Cache dir "{cache_dir}" already exists! Either remove '
                    f'it or set reuse=True.'
                )
        # eviction_policy='none' deactivates the cache size limit (of 1GB by
        # default)
        self.cache = diskcache.Cache(cache_dir, eviction_policy='none')

    def __getitem__(self, item):
        return self.cache[item]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, item):
        return item in self.cache

    def __len__(self):
        return len(self.cache)

    def __del__(self):
        # This gets called when all references to the cache wrapper are
        # dropped and the program is still running. This includes normal
        # termination and keyboard interrupt, but no other signals like
        # SIGTERM or SIGKILL. Some signals sometimes work if they are handled
        # within python.
        self.cache.close()
        if self.clear:
            if Path(self.cache.directory).exists():
                import shutil
                shutil.rmtree(self.cache.directory)


class DiskCacheDataset(CacheDataset):
    """
    We use the `diskcache` package because it provides a simple interface and
    is thread-safe and forkable. This means it works with all backends for
    prefetching.
    """
    def __init__(self, input_dataset, cache_dir=None, reuse=True, clear=True):
        # We have the same assumptions as CacheDataset
        self.input_dataset = input_dataset
        assert self.input_dataset.indexable, self.input_dataset
        self._cache = _DiskCacheWrapper(cache_dir, reuse, clear)

    def copy(self, freeze: bool = False) -> 'Dataset':
        if not freeze:
            import warnings
            warnings.warn(
                'Copying a CacheDataset preserves the cache, i.e., the '
                'already cached part of the dataset will be frozen even if '
                'freeze=False!'
            )
        # We have to share the cache here because otherwise a new cache would
        # be initialized at every copy and copy is called by prefetch before
        # iterating over the dataset
        copy = self.__class__.__new__(self.__class__)
        copy.input_dataset = self.input_dataset.copy(freeze)
        copy._cache = self._cache
        return copy

    def check(self):
        import shutil
        import humanfriendly
        diskusage = shutil.disk_usage(self._cache.cache.directory)
        if diskusage.free < 5 * 1024 ** 3:
            import warnings
            warnings.warn(
                f'There is not much space left in the specified cache '
                f'dir "{self._cache.cache.directory}"! (total='
                f'{humanfriendly.format_size(diskusage.total, binary=True)}'
                f', free='
                f'{humanfriendly.format_size(diskusage.free, binary=True)}'
                f')', ResourceWarning
            )
            if diskusage.free < 1 * 1024 ** 3:
                # Crash if less than 1GB left. It's better to crash
                # this process than to crash the whole machine
                raise RuntimeError(
                    f'Not enough space on device! The device that the '
                    f'cache directory "{self._cache.cache.directory}" '
                    f'is located on has less than 1GB '
                    f'space left. You probably want to delete some '
                    f'files before crashing the machine.'
                )
        return True

    def __str__(self):
        return (
            f'{self.__class__.__name__}(cache_dir='
            f'{self._cache.cache.directory}, reuse={self._cache.reuse})'
        )


class ProfilingDataset(Dataset):
    """
    Special dataset to profile each dataset.
    The repr will be modified to display the time that was spend in a
    particular part of the pipeline.
    
    Warning:
        This Dataset has limited support for prefetching. Only GIL bounded
        prefetching is supported (e.g. thread pool and not process pool).

    Example:
        >>> import lazy_dataset
        >>> from lazy_dataset.core import ProfilingDataset
        >>> ds = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3})
        >>> def sleep(ex):
        ...     time.sleep(1)
        ...     return ex
        >>> ds_copy = ProfilingDataset(ds.map(sleep))
        >>> _ = list(ds_copy)
        >>> print(repr(ds_copy))  # doctest: +ELLIPSIS
            DictDataset(len=3) (fetch duration = 0:00:00.000..., hits = 3)
          MapDataset(_pickle.loads) (fetch duration = 0:00:00.000..., hits = 3)
        MapDataset(<function sleep at 0x...>) (fetch duration = 0:00:03.00..., hits = 3)
        >>> ds_copy = ProfilingDataset(ds.map(sleep).prefetch(4, 8))
        >>> _ = list(ds_copy)
        >>> print(repr(ds_copy))  # doctest: +ELLIPSIS
              DictDataset(len=3) (fetch duration = 0:00:00.000..., hits = 3)
            MapDataset(_pickle.loads) (fetch duration = 0:00:00.000..., hits = 3)
          MapDataset(<function sleep at 0x...>) (fetch duration = 0:00:03.00..., hits = 3)
        PrefetchDataset(4, 8, 't') (fetch duration = 0:00:01.0..., hits = 3)
        >>> def f(ex):
        ...     if ex % 2 == 0:
        ...         raise FilterException()
        ...     return ex
        >>> ds_copy = ProfilingDataset(ds.map(f).map(sleep).catch(FilterException))
        >>> _ = list(ds_copy)
        >>> print(repr(ds_copy))     # doctest: +ELLIPSIS
                DictDataset(len=3) (fetch duration = 0:00:00.0..., hits = 3)
              MapDataset(_pickle.loads) (fetch duration = 0:00:00.0..., hits = 3)
            MapDataset(<function f at 0x...>) (fetch duration = 0:00:00.0..., hits = 3 (1 filtered))
          MapDataset(<function sleep at 0x...>) (fetch duration = 0:00:02.0..., hits = 3 (1 filtered))
        CatchExceptionDataset() (fetch duration = 0:00:02.0..., hits = 2)

        >>> ds_copy = ProfilingDataset(ds.map(f).items().catch(FilterException))
        >>> _ = list(ds_copy)
        >>> print(repr(ds_copy))  # doctest: +ELLIPSIS
                DictDataset(len=3) (fetch duration = 0:00:00.00..., hits = 3)
              MapDataset(_pickle.loads) (fetch duration = 0:00:00.00..., hits = 3)
            MapDataset(<function f at 0x...>) (fetch duration = 0:00:00.00..., hits = 3 (1 filtered))
          ItemsDataset() (fetch duration = 0:00:00.00..., hits = 3 (1 filtered))
        CatchExceptionDataset() (fetch duration = 0:00:00.00..., hits = 2)
    """
    # alternative time.process_time
    timestamp = staticmethod(time.perf_counter)

    def __init__(self, input_dataset):
        if isinstance(input_dataset, self.__class__):
            # This is not necessary, but the output is difficult to interpret.
            raise RuntimeError(
                'You can use ProfilingDataset only once.\n'
                'The input_dataset is already a ProfilingDataset:\n'
                f'{repr(input_dataset)}'
            )
        input_dataset = input_dataset.copy()

        # use list with one element as mutable container to share the timer
        # between the copies (Necessary for prefetch)
        self.time = [0.0]
        # Use list with two elements as mutable container for hit count to share
        # it between copies: [<total_hit_count>, <failed_hit_count>]
        self.hit_count = [0, 0]

        if hasattr(input_dataset, 'input_datasets'):
            input_dataset.input_datasets = [
                self.__class__(ds)
                for ds in input_dataset.input_datasets
            ]
        if hasattr(input_dataset, 'input_dataset'):
            input_dataset.input_dataset = ProfilingDataset(
                input_dataset.input_dataset)

        self.input_dataset = input_dataset

    def __repr__(self):
        r = repr(self.input_dataset)

        hits = f'hits = {self.hit_count[0]}'
        if self.hit_count[1]:
            # Better alternative for the name "filtered"?
            hits += f' ({self.hit_count[1]} filtered)'

        # Better alternative for the name "fetch duration"?
        r += (f' (fetch duration = {datetime.timedelta(seconds=self.time[0])}, '
              f'{hits})')
        return r

    def __len__(self):
        return len(self.input_dataset)

    def indexable(self):
        return self.input_dataset.indexable()

    def keys(self):
        return self.input_dataset.keys()

    def __iter__(self, with_key=False):
        if with_key:
            raise _ItemsNotDefined(self.__class__.__name__)
        it = iter(self.input_dataset)
        while True:
            start = self.timestamp()
            self.hit_count[0] += 1
            try:
                x = next(it)
            except StopIteration:
                self.hit_count[0] -= 1
                return
            except Exception:
                self.hit_count[1] += 1
                raise
            finally:
                end = self.timestamp()
                self.time[0] += (end - start)
            yield x

    def __getitem__(self, item):
        start = self.timestamp()
        # Avoid context manager: https://stackoverflow.com/a/26156031/5766934
        self.hit_count[0] += 1
        try:
            return self.input_dataset[item]
        except Exception:
            self.hit_count[1] += 1
            raise
        finally:
            end = self.timestamp()
            self.time[0] += (end - start)

    def copy(self, freeze=False):
        # Use __new__ to disable the copy in the __init__
        new = self.__class__.__new__(self.__class__)
        new.input_dataset = self.input_dataset.copy(freeze=freeze)
        # Share time for prefetch
        new.time = self.time
        new.hit_count = self.hit_count
        return new


if __name__ == '__main__':
    import doctest
    doctest.testmod()
