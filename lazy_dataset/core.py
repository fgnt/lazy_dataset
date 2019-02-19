"""
The reader is part of the new database concept 2017.

The task of the reader is to take a database JSON and an dataset identifier as
an input and load all meta data for each observation with corresponding
numpy arrays for each time signal (non stacked).

An example ID often stands for utterance ID. In case of speaker mixtures,
it replaces mixture ID. Also, in case of event detection utterance ID is not
very adequate.

The JSON file is specified as follows:

datasets:
    <dataset name 0>
        <unique example id 1> (unique within a dataset)
            audio_path:
                speech_source:
                    <path to speech of speaker 0>
                    <path to speech of speaker 1>
                observation:
                    blue_array: (a list, since there are no missing channels)
                        <path to observation of blue_array and channel 0>
                        <path to observation of blue_array and channel 0>
                        ...
                    red_array: (special case for missing channels)
                        c0: <path to observation of red_array and channel 0>
                        c99: <path to observation of red_array and channel 99>
                        ...
                speech_image:
                    ...
            speaker_id:
                <speaker_id for speaker 0>
                ...
            gender:
                <m/f>
                ...
            ...

Make sure, all keys are natsorted in the JSON file.

Make sure, the names are not redundant and it is clear, which is train, dev and
test set. The names should be as close as possible to the original database
names.

An observation/ example has information according to the keys file.

If a database does not have different arrays, the array dimension can be
omitted. Same holds true for the channel axis or the speaker axis.

The different axis have to be natsorted, when they are converted to numpy
arrays. Skipping numbers (i.e. c0, c99) is database specific and is not handled
by a generic implementation.

If audio paths are a list, they will be stacked to a numpy array. If it is a
dictionary, it will become a dictionary of numpy arrays.

If the example IDs are not unique in the original database, the example IDs
are made unique by prefixing them with the dataset name of the original
database, i.e. dt_simu_c0123.
"""
import logging
import numbers
import textwrap
import operator
from copy import deepcopy
from pathlib import Path
import itertools
import random as rnd

from cached_property import cached_property
import numpy as np

from paderbox import kaldi
from paderbox.database import keys
from paderbox.io.audioread import audioread

LOG = logging.getLogger('Database')

import collections


class FilterException(Exception):
    """
    Special Exception for the Dataset to indicate that this example should be
    skipped. The `Dataset.catch()` and
    `Dataset.prefetch(..., catch_filter_exception=True)` handle this
    exception.
    """
    pass


class Dataset:
    def __call__(self):
        """
        Usecase
          tf.data.Dataset.from_generator(dataset)
        Without __call__:
          tf.data.Dataset.from_generator(lambda: dataset)
        """
        return self.__iter__()

    def __iter__(self):
        raise NotImplementedError(
            f'__iter__ is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def __len__(self):
        # The correct exception type is TypeError and not NotImplementedError
        # for __len__. For example len(dataset) ignores TypeError but not
        # NotImplementedError
        raise TypeError(
            f'__len__ is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def __getitem__(self, item):
        if isinstance(item, (slice, tuple, list)):
            return SliceDataset(item, self)
        elif isinstance(item, np.ndarray) and item.ndim == 1:
            return SliceDataset(item, self)
        elif isinstance(item, bytes):
            raise NotImplementedError(
                f'This is not implemented for an bytes objext. Use bytes.decode() to convert it to an str.\n'
                f'__getitem__ is not implemented for {self.__class__}[{item!r}],\n'
                f'where type({item!r}) == {type(item)} '
                f'self: \n{self!r}'
            )
        raise NotImplementedError(
            f'__getitem__ is not implemented for {self.__class__}[{item!r}],\n'
            f'where type({item!r}) == {type(item)}\n'
            f'self:\n{self!r}'
        )

    def keys(self):
        raise NotImplementedError(
            f'keys is not implemented for {self.__class__}.\n'
            f'self: \n{repr(self)}'
        )

    def items(self):
        """
        >>> examples = {'a': {'d': 1}, 'b': {'e': 1}, 'c': {'f': 1}}
        >>> it = DictDataset(examples)
        >>> list(it)
        [{'d': 1}, {'e': 1}, {'f': 1}]
        >>> list(it.items())
        [('a', {'d': 1}), ('b', {'e': 1}), ('c', {'f': 1})]
        """
        it = DictDataset(dict(zip(self.keys(), self.keys())))
        return it.zip(self)

    def __contains__(self, item):
        # contains is not well defined for dataset, because dataset is a
        # mixture of tuple and dict. (tuple -> value, dict -> key)
        # Use the verbose contains (see exception msg)
        raise Exception(
            f"Use 'key in {self.__class__}.keys()' "
            f"instead of 'key in {self.__class__}'")

    def map(self, map_fn, num_workers=0, buffer_size=100, backend='t'):
        """
        :param map_fn: function to transform an example dict. Takes an example
            dict as provided by this dataset and returns a transformed
            example dict, e.g. read and adss the observed audio signals.
        :param num_workers:
        :param buffer_size:
        :return: MapDataset returning mapped examples. This can e.g. be
        used to read and add audio to the example dict (see read_audio method).

        Note:
          - map_fn can do inplace transformations without using copy.
            The ExampleDataset makes a deepcopy of each example and prevents a
            modification of the root example.
          - If num_workers > 0 that the map_fn is performed in parallel.
            But the input dataset is still executed serial.
            This allows an arbitrary input dataset. When it is desired to get
            an example in parallel, use prefetch on an indexable dataset.
        """
        if num_workers > 0:
            return ParMapDataset(
                map_fn, self, num_workers=num_workers, buffer_size=buffer_size,
                backend=backend
            )
        return MapDataset(map_fn, self)

    def prefetch(self, num_workers, buffer_size, backend='t', catch_filter_exception=None):
        """

        Args:
            num_workers:
            buffer_size:

        Returns:

        >>> import string
        >>> ascii = string.ascii_lowercase
        >>> it = DictDataset({k: v for v, k in enumerate(ascii[:10])})
        >>> # ds1 = ds1.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> list(it)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> def foo(ex):
        ...     print(f'called with {ex}')
        ...     return ex
        >>> it = it.map(foo)
        >>> list(it)
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
        >>> it = it.prefetch(2, 4)
        >>> next(iter(it))
        called with 0
        called with 1
        called with 2
        called with 3
        0

        """
        return PrefetchDataset(
            input_dataset=self,
            num_workers=num_workers,
            buffer_size=buffer_size,
            backend=backend,
            catch_filter_exception=catch_filter_exception,
        )

    def filter(self, filter_fn, lazy=True):
        """
        The filter_fn consumes an example. If the filter_fn returns True, we
        keep the example. If it is False, we drop the example.

        Filtering examples. If using lazy=False this method should be called
        before applying expensive map functions.

        Syntax is inspired by:
        https://docs.python.org/3/library/functions.html#filter

        :param filter_fn: function to filter examples, takes example as input
            and returns True if example should be kept, else False.
        :param lazy: If True, dataset does not support `len(it)` anymore but
            computation is performed once the dataset visits the item.
        :return: FilterDataset iterating over filtered examples.
        """
        if lazy:
            # Input dataset can be indexable, but this is not needed.
            # Output still does not have `len` and is not indexable.
            return FilterDataset(filter_fn, self)
        else:
            # Input dataset needs to be indexable.
            # Output still has `len`, following line should not raise errors.
            try:
                _ = self.keys()
            except Exception:
                raise RuntimeError(
                    'You can only use lazy=False if the incoming dataset is '
                    'indexable.'
                )
            return self[[i for i, e in enumerate(self) if filter_fn(e)]]

    def catch(self, exceptions=FilterException, warn=False):
        """
        Drop examples that throw an exception (default: FilterException).
        This is an alternative to filter.

        Args:
            exceptions:
            warn: If True, enable logger warning.

        Returns:
        """
        return CatchExceptionDataset(self, exceptions=exceptions, warn=warn)

    def concatenate(self, *others):
        """
        Concatenate this dataset with others. keys need to be unambiguous.
        :param others: list of other datasets to be concatenated
        :return: ExamplesDataset iterating over all examples.
        """
        if len(others) == 0:
            return self
        return ConcatenateDataset(self, *others)

    def zip(self, *others):
        """
        Creates a `Dataset` by zipping together the given datasets.

        This method has two major differences to the built-in `zip()` function
        in Python. First the zipping happen based on the keys of the
        first dataset (i.e. The first defines the order).

        Second it is assumes that all datasets have the same length and keys.
        (Could be removed, when someone needs it.)

        This function is usually followed by a map call to merge the tuple of
        dicts to a single dict.

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
                ExamplesDataset(len=2)
                ExamplesDataset(len=2)
              ZipDataset()
            MapDataset(<function <lambda> at 0x...>)
                ExamplesDataset(len=2)
                ExamplesDataset(len=2)
              ZipDataset()
            MapDataset(<function <lambda> at 0x...>)
          ZipDataset()
        MapDataset(<function <lambda> at 0x...>)
        >>> for e in ds4: print(e)
        {'example_id': 'a', 'z': 1, 'y': 'c'}
        {'example_id': 'b', 'z': 3, 'y': 'd'}

        # Lambda that merges an arbitary amount of dicts.
        >>> ds5 = ds3.map(lambda example: dict(sum([list(e.items()) for e in example], [])))
        >>> for e in ds5: print(e)
        {'example_id': 'a', 'z': 1, 'y': 'c'}
        {'example_id': 'b', 'z': 3, 'y': 'd'}

        :param others: list of other datasets to be zipped
        :return: Dataset
        """
        return ZipDataset(self, *others)

    def shuffle(self, reshuffle=False, rng=None, buffer_size=None):
        """
        Shuffle this dataset.
        :param reshuffle:
            If True, shuffle on each iteration, but disable indexing.
            If False, single shuffle, but support indexing.
        :param rng:
        :param buffer_size:
        :return:

        Note:
         - Use the buffer_size only in special cases were the dataset is
           already shuffled. For example a dataset is shuffled and then a
           fragment call splits on into multiple examples. In this case a local
           shuffle (i.e. buffer_size > 0) is reasonable.

        """
        # Should reshuffle default be True or False
        if buffer_size is not None:
            assert reshuffle is True, 'LocalShuffleDataset only supports reshuffle'
            assert rng is None, 'LocalShuffleDataset does not support seeds.'
            return LocalShuffleDataset(self, buffer_size=buffer_size)
        if reshuffle is True:
            assert rng is None, 'ReShuffleDataset does not support seeds.'
            return ReShuffleDataset(self)
        elif reshuffle is False:
            return ShuffleDataset(self, rng=rng)
        else:
            raise ValueError(reshuffle, self)

    def tile(self, reps, shuffle=False):
        """
        Constructs an new dataset by repeating the dataset the number of
        times given by reps.

        The shuffle option if provided, because before concatenating the
        shuffle is applied.

        """
        datasets = [self] * reps
        if shuffle:
            datasets = [
                it.shuffle()
                for it in datasets
            ]
        return self.__class__.concatenate(*datasets)

    def groupby(self, group_fn):
        """
        >>> from IPython.lib.pretty import pprint
        >>> examples = {'a': {'z': 1}, 'b': {'z': 2}, 'c': {'z': 1}, 'd': {'z': 1}, 'e': {'z': 3}}
        >>> it = DictDataset(examples)
        >>> for k, v in it.groupby(lambda ex: ex['z']).items():
        ...     print(f'{k}:', list(v), v.keys())
        ...     print(f'{v!r}')
        1: [{'z': 1}, {'z': 1}, {'z': 1}] ('a', 'c', 'd')
          ExamplesDataset(len=5)
        SliceDataset([0, 2, 3])
        2: [{'z': 2}] ('b',)
          ExamplesDataset(len=5)
        SliceDataset([1])
        3: [{'z': 3}] ('e',)
          ExamplesDataset(len=5)
        SliceDataset([4])
        """
        iterable = enumerate(list(self.map(group_fn)))
        groups = collections.defaultdict(list)
        for k, g in itertools.groupby(iterable, lambda ele: ele[1]):
            indices = [ele[0] for ele in g]
            groups[k] += indices
        return {k: self[v] for k, v in groups.items()}

    def split(self, sections):
        """
        >>> examples = {'a': {}, 'b': {}, 'c': {}, 'd': {}, 'e': {}}
        >>> it = DictDataset(examples)
        >>> it = it.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> its = it.split(2)
        >>> list(its[0])
        [{'example_id': 'a'}, {'example_id': 'b'}, {'example_id': 'c'}]
        >>> list(its[1])
        [{'example_id': 'd'}, {'example_id': 'e'}]
        >>> list(its[1].keys())
        """
        if sections < 1:
            raise ValueError("sections must be >= 1")
        if sections > len(self):
            raise ValueError(
                f'Dataset has only {len(self)} elements and cannot be '
                f'split into {sections} sections.'
            )
        slices = np.array_split(np.arange(len(self)), sections)
        return [self[list(s)] for s in slices]

    def fragment(self, fragment_fn):
        """
        Fragments each example into multiple new examples.
        E.g. use channels as single examples or split each example into segments

        >>> examples = {'a': [1, 2], 'b': [3, 4]}
        >>> it = DictDataset(examples)
        >>> list(it)
        [[1, 2], [3, 4]]
        >>> list(it.fragment(lambda ex: ex))
        [1, 2, 3, 4]
        """
        return FragmentDataset(fragment_fn, self)

    def sort(self, key_fn, sort_fn=sorted):
        """
        Sorts the dataset with the entry described by key_list
        >>> examples = {'a': {'x': 1}, 'b': {'x': 3},  'c': {'x': 12}, 'd': {'x': 2}}
        >>> it = DictDataset(examples)
        >>> it_sorted = it.sort(lambda ex: ex['x'])
        >>> it_sorted
          ExamplesDataset(len=4)
        SliceDataset(('a', 'd', 'b', 'c'))
        >>> print(it_sorted.slice)
        (0, 3, 1, 2)
        >>> list(it_sorted)
        [{'x': 1}, {'x': 2}, {'x': 3}, {'x': 12}]
        """
        sort_values = [key_fn(self[key]) for key in self.keys()]
        return self[tuple([key for _, key in
                           sort_fn(zip(sort_values, self.keys()))])]

    def shard(self, num_shards, shard_index):
        """
        Splits an dataset into `num_shards` shards and
        selects shard `shard_index`.
        """
        return self.split(num_shards)[shard_index]

    def batch(self, batch_size, drop_last=False):
        """
        :param batch_size:
        :param drop_last:
        :return:
        """
        return BatchDataset(self, batch_size, drop_last)

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
            for input_dataset in self.input_datasets:
                s = repr(input_dataset)
                r += textwrap.indent(s, indent) + '\n'
        return r + str(self)

    def random_choice(
            self,
            size=None,
            replace=False,
            rng_state: np.random.RandomState=np.random,
    ):
        """
        >>> rng_state = np.random.RandomState(0)
        >>> examples = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
        >>> it = DictDataset(examples)
        >>> def foo(ex):
        ...     print('foo')
        ...     return ex
        >>> it = it.map(foo)
        >>> print('Number', it.random_choice(rng_state=rng_state))
        foo
        Number 3

        >>> print(it.random_choice(1, rng_state=rng_state))
        SliceDataset([0])
        >>> print(it.random_choice(2, rng_state=rng_state))
        SliceDataset([1 3])
        >>> it_choice = it.random_choice(7, rng_state=rng_state, replace=True)
        >>> print(it_choice)
        SliceDataset([0 4 2 1 0 1 1])
        >>> print(list(it_choice))
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

    def apply(self, apply_fn: callable):
        """Allows to apply functions to the complete dataset, not to the
        examples itself.

        Args:
            apply_fn: For now, it is a single function, e.g.
                `lambda it: it.shard(num_shards, shard_index)` but can
                potentially be a list in future implementations.

        Returns:

        """
        if apply_fn is None:
            return self
        elif isinstance(apply_fn, list):
            raise NotImplementedError
        else:
            return apply_fn(self)


class DictDataset(Dataset):
    """
    Dataset to iterate over a list of examples with each example being a dict
    according to the json structure as outline in the top of this file.
    """

    def __init__(self, examples, name=None):
        assert isinstance(examples, dict)
        self.examples = examples
        self.name = name
        self._keys = tuple(self.examples.keys())

    def __str__(self):
        if self.name is None:
            return f'{self.__class__.__name__}(len={len(self)})'
        else:
            return f'{self.__class__.__name__}' \
                   f'(name={self.name}, len={len(self)})'

    def keys(self):
        return self._keys

    def __iter__(self):
        for k in self.keys():
            yield self[k]

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                example = self.examples[item]
            except Exception:
                import difflib
                similar = difflib.get_close_matches(item, self.keys())
                raise KeyError(item, f'close_matches: {similar}', self)
        elif isinstance(item, numbers.Integral):
            key = self.keys()[item]
            example = self.examples[key]
        else:
            return super().__getitem__(item)

        # Ensure that nobody can change the data.
        return deepcopy(example)

    def __len__(self):
        return len(self.examples)


class MapDataset(Dataset):
    """
    Dataset that iterates over an input_dataset and applies a transformation
    map_function to each element.

    """

    def __init__(self, map_function, input_dataset):
        """

        :param map_function: function that transforms an element of
            input_dataset. Use deepcopy within the map_function if necessary.
        :param input_dataset: any dataset (e.g. ExampleDataset)
        """
        assert callable(map_function), map_function
        self.map_function = map_function
        self.input_dataset = input_dataset

    def __str__(self):
        return f'{self.__class__.__name__}({self.map_function})'

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        for example in self.input_dataset:
            yield self.map_function(example)

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

    def __iter__(self):

        from paderbox.utils.parallel_utils import lazy_parallel_map

        return lazy_parallel_map(
            self.map_function,
            self.input_dataset,
            buffer_size=self.buffer_size,
            max_workers=self.num_workers,
            backend=self.backend,
        )


class CatchExceptionDataset(Dataset):
    """
    >>> it = DictDataset({'a': 1, 'b': 2, 'c': 3})
    >>> list(it)
    [1, 2, 3]
    >>> def foo(integer):
    ...     if integer == 2:
    ...         raise FilterException('Exception msg')
    ...     else:
    ...         return integer
    >>> list(it.map(foo))
    Traceback (most recent call last):
    ...
    dataset.FilterException: Exception msg
    >>> list(it.map(foo).catch())
    [1, 3]
    >>> it.map(foo).catch()[0]  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    NotImplementedError: __getitem__ is not well defined for <class 'dataset.CatchExceptionDataset'>[0],
    because 0 is an index
    self:
        ExamplesDataset(len=3)
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

    def __iter__(self):
        for i in range(len(self.input_dataset)):
            try:
                yield self.input_dataset[i]
            except self.exceptions as e:
                if self.warn:
                    msg = repr(e)
                    LOG.warning(msg)


class PrefetchDataset(Dataset):
    def __init__(
            self,
            input_dataset,
            num_workers,
            buffer_size,
            backend='t',
            catch_filter_exception=False,
    ):

        # Input dataset needs to be indexable.
        try:
            _ = len(input_dataset)
        except Exception:
            raise RuntimeError(
                'You can only use Prefetch if the incoming dataset is '
                'indexable.'
            )
        assert num_workers >= 1, num_workers
        assert buffer_size >= num_workers, (num_workers, buffer_size)

        self.input_dataset = input_dataset
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.backend = backend
        self.catch_filter_exception = catch_filter_exception

    def __iter__(self):

        from paderbox.utils.parallel_utils import lazy_parallel_map

        if (
                self.catch_filter_exception is False
                or self.catch_filter_exception is None
                or (
                    isinstance(self.catch_filter_exception, (tuple, list))
                    and len(self.catch_filter_exception) == 0
                )
        ):
            yield from lazy_parallel_map(
                self.input_dataset.__getitem__,
                range(len(self.input_dataset)),
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

            def catcher(index):
                try:
                    return self.input_dataset[index]
                except catch_filter_exception:
                    return unique_object

            for data in lazy_parallel_map(
                catcher,
                range(len(self.input_dataset)),
                buffer_size=self.buffer_size,
                max_workers=self.num_workers,
                backend=self.backend,
            ):
                if data is unique_object:
                    pass
                else:
                    yield data

    def __str__(self):
        return (
            f'{self.__class__.__name__}'
            f'({self.num_workers}, {self.buffer_size}, {self.backend!r})'
        )


class ShuffleDataset(Dataset):
    """
    Dataset that shuffles the input_dataset. Assumes, that the input_dataset
    has a length.
    Note:
        This Dataset supports indexing, but does not reshuffle each iteration.

    >>> np.random.seed(1)
    >>> examples = {'a': {}, 'b': {}, 'c': {}}
    >>> it = DictDataset(examples)
    >>> it = it.items().map(lambda x: {'example_id': x[0], **x[1]})
    >>> it = it.shuffle(False)
    >>> it  # doctest: +ELLIPSIS
          ExamplesDataset(len=3)
          ExamplesDataset(len=3)
        ZipDataset()
      MapDataset(<function <lambda> at 0x...>)
    ShuffleDataset()
    >>> list(it)
    [{'example_id': 'a'}, {'example_id': 'c'}, {'example_id': 'b'}]
    >>> it.keys()
    ('a', 'c', 'b')
    """

    def __init__(self, input_dataset, rng=None):
        self.permutation = np.arange(len(input_dataset))
        rng = np.random if rng is None else rng
        rng.shuffle(self.permutation)
        self.input_dataset = input_dataset

    def __len__(self):
        return len(self.input_dataset)

    _keys = None

    def keys(self):
        if self._keys is None:
            keys = self.input_dataset.keys()
            self._keys = tuple([keys[p] for p in self.permutation])
        return self._keys

    def __iter__(self):
        for idx in self.permutation:
            yield self.input_dataset[idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_dataset[item]
        elif isinstance(item, numbers.Integral):
            return self.input_dataset[self.permutation[item]]
        else:
            return super().__getitem__(item)


class ReShuffleDataset(Dataset):
    """
    Dataset that shuffles the input_dataset. Assumes, that the input_dataset
    has a length.
    Note:
        This Dataset reshuffle each iteration, but does not support indexing.
    """

    def __init__(self, input_dataset):
        self.permutation = np.arange(len(input_dataset))
        self.input_dataset = input_dataset

    def __len__(self):
        return len(self.input_dataset)

    # keys is not well defined for this dataset
    # The First dataset (i.e. ExamplesDataset has sorted keys), so what should
    # this dataset return? Maybe a frozenset to highlight unordered?
    # def keys(self):
    #     return frozenset(self.input_dataset.keys())

    def __iter__(self):
        np.random.shuffle(self.permutation)
        for idx in self.permutation:
            yield self.input_dataset[idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_dataset[item]
        elif isinstance(item, (numbers.Integral, slice, tuple, list)):
            raise TypeError(
                f'{self.__class__.__name__} does not support '
                f'integers and slices as argument of __getitem__.'
                f'Got argument "{item}" of type {type(item)}.'
            )
        else:
            return super().__getitem__(item)


class LocalShuffleDataset(Dataset):
    """
    Dataset that shuffles the input_dataset locally by randomly sampling from
    a fixed length buffer. Hence also applicable to Datasets that does not
    support indexing
    Note:
        This Dataset reshuffles each iteration, but does not support indexing.
    """

    def __init__(self, input_dataset, buffer_size=100):
        self.input_dataset = input_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        buffer = list()
        print(f'Filling Shuffle Buffer with {self.buffer_size} samples.')
        buffer_filled = False
        for element in self.input_dataset:
            buffer.append(element)
            if len(buffer) >= self.buffer_size:
                if not buffer_filled:
                    print('Shuffle Buffer filled.')
                    buffer_filled = True
                yield buffer.pop(int(np.random.choice(self.buffer_size)))
        rnd.shuffle(buffer)
        for element in buffer:
            yield element

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.input_dataset[item]
        elif isinstance(item, (numbers.Integral, slice, tuple, list)):
            raise TypeError(
                f'{self.__class__.__name__} does not support '
                f'integers and slices as argument of __getitem__.'
                f'Got argument "{item}" of type {type(item)}.'
            )
        else:
            return super().__getitem__(item)


class SliceDataset(Dataset):
    def __init__(self, slice, input_dataset):
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
        self._slice = slice
        try:
            self.slice = np.arange(len(input_dataset))[self._slice]
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

    _keys = None

    def keys(self):
        if self._keys is None:
            keys = self.input_dataset.keys()
            # itemgetter makes the same as
            # "tuple([keys[i] for i in self.slice])"
            # but is 10 times faster
            self._keys = operator.itemgetter(*self.slice)(keys)
        return self._keys

    def __len__(self):
        return len(self.slice)

    def __str__(self):
        if isinstance(self._slice, (tuple, list)):
            slice_str = textwrap.shorten(
                str(self._slice)[1:-1],
                width=50,
                placeholder=' ...',
            )
            slice_str = f'[{slice_str}]'
        else:
            slice_str = str(self._slice)

        return f'{self.__class__.__name__}({slice_str})'

    def __iter__(self):
        for idx in self.slice:
            yield self.input_dataset[idx]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.input_dataset[key]
        elif isinstance(key, numbers.Integral):
            return self.input_dataset[self.slice[key]]
        else:
            return super().__getitem__(key)


class FilterDataset(Dataset):
    """
    Dataset that iterates only over those elements of input_dataset that meet
    filter_function.
    """

    def __init__(self, filter_function, input_dataset):
        """

        :param filter_function: a function that takes an element of the input
            dataset and returns True if the element is valid else False.
        :param input_dataset: any dataset (e.g. ExampleDataset)
        """
        assert callable(filter_function), filter_function
        self.filter_function = filter_function
        self.input_dataset = input_dataset

    def __str__(self):
        return f'{self.__class__.__name__}({self.filter_function})'

    def __iter__(self):
        for example in self.input_dataset:
            if self.filter_function(example):
                yield example

    def __getitem__(self, key):
        assert isinstance(key, str), (
            f'key == {key!r}\n{self.__class__} does not support __getitem__ '
            f'for type(key) == {type(key)},\n'
            f'Only type str is allowed.\n'
            f'self:\n{repr(self)}'
        )
        ex = self.input_dataset[key]
        if not self.filter_function(ex):
            raise IndexError(key)
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
        :param input_datasets: list of datasets
        """
        self.input_datasets = input_datasets

    def __iter__(self):
        for input_dataset in self.input_datasets:
            for example in input_dataset:
                yield example

    def __len__(self):
        return sum([len(i) for i in self.input_datasets])

    _keys = None

    def keys(self):
        """
        >>> examples = {'a': 1, 'b': 2, 'c': 3}
        >>> it = DictDataset(examples)
        >>> it.concatenate(it).keys()
        Traceback (most recent call last):
        ...
        AssertionError: Keys are not unique. There are 3 duplicates.
        ['a', 'b', 'c']
        >>> list(it.concatenate(it.map(lambda x: x+10)))
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

            assert len(keys) == len(set(keys)), \
                'Keys are not unique. ' \
                'len(self._keys) = {len(self._keys)} != ' \
                '{len(set(self._keys))} = len(set(self._keys))'
            self._keys = tuple(keys)
        return self._keys

    def __getitem__(self, item):
        """
        >>> it1 = DictDataset({'a': {}, 'b': {}})
        >>> it1 = it1.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> it2 = DictDataset({'c': {}, 'd': {}})
        >>> it2 = it2.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> it = it1.concatenate(it2)
        >>> it['a']
        {'example_id': 'a'}
        >>> it['c']
        {'example_id': 'c'}
        """
        if isinstance(item, numbers.Integral):
            for dataset in self.input_datasets:
                if len(dataset) <= item:
                    item -= len(dataset)
                else:
                    return dataset[item]
            raise KeyError(item)
        elif isinstance(item, str):
            self.keys()  # test unique keys
            for dataset in self.input_datasets:
                if item in dataset.keys():
                    return dataset[item]
            # In collections.ChainMap is
            # 'try: ... except KeyError: ...'
            # used, since an dataset should provide a better exception msg,
            # __contains__ is faster than collections.ChainMap
            # because the overhead of calculating the exception msg is to high.

            if item in self.keys():
                raise Exception(
                    f'There is a internal error in {self.__class__}. '
                    f'Could not find {item} in input datasets, but it is in '
                    f'{self.keys()}'
                )
            raise KeyError(item)
        else:
            return super().__getitem__(item)


class ZipDataset(Dataset):
    """
    See Dataset.zip
    """

    def __init__(self, *input_datasets):
        """
        :param input_datasets: list of datasets
        """
        self.input_datasets = input_datasets
        assert len(self.input_datasets) >= 1, \
            'You have to provide at least one dataset.' \
            f'\n{self.input_datasets}'
        assert len(self.input_datasets) >= 2, \
            'Currently limited to at least two dataset. Could be removed.' \
            f'\n{self.input_datasets}'
        lengths = [len(it) for it in self.input_datasets]
        keys = set(self.input_datasets[0].keys())
        lengths = [
            len(keys - set(it.keys())) for it in self.input_datasets
        ]
        if set(lengths) != {0}:
            missing_keys = [
                keys - set(it.keys()) for it in self.input_datasets
            ]
            raise AssertionError(
                f'Expect that all input_datasets have at least the same keys as ' \
                f'the first. To much keys: {missing_keys}' \
                f'\n{self.input_datasets}'
            )

    def __iter__(self):
        for key in self.keys():
            yield tuple([
                it[key]
                for it in self.input_datasets
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
                it[item]
                for it in self.input_datasets
            ])
        else:
            return super().__getitem__(item)


class FragmentDataset(Dataset):
    """
    Fragments each example from an input_dataset into multiple new examples.
    E.g. use channels as single examples or split each example into segments
    """
    def __init__(self, fragment_fn, input_dataset):
        self.fragment_fn = fragment_fn
        self.input_dataset = input_dataset

    def __iter__(self):
        for example in self.input_dataset:
            for fragment in self.fragment_fn(example):
                yield fragment


class BatchDataset(Dataset):
    """

    >>> from paderbox.database.dataset import ExamplesDataset
    >>> import string
    >>> examples = {c: i for i, c in enumerate(string.ascii_letters[:7])}
    >>> it = DictDataset(examples)
    >>> it = it.batch(3)
    >>> it
      ExamplesDataset(len=7)
    BatchDataset(batch_size=3)
    >>> list(it), len(it)
    ([[0, 1, 2], [3, 4, 5], [6]], 3)
    >>> it[2], it[-1]
    ([6], [6])
    >>> it[3]
    Traceback (most recent call last):
    ...
    IndexError: tuple index out of range
    >>> it = DictDataset(examples)
    >>> it = it.batch(3, drop_last=True)
    >>> list(it), len(it)
    ([[0, 1, 2], [3, 4, 5]], 2)
    >>> it[-1]
    [3, 4, 5]
    >>> it = DictDataset(examples)[:6]
    >>> it = it.batch(3)
    >>> list(it), len(it)
    ([[0, 1, 2], [3, 4, 5]], 2)
    >>> it[1]
    [3, 4, 5]
    >>> it['abc']
    Traceback (most recent call last):
    ...
    NotImplementedError: __getitem__ is not implemented for <class 'nt.database.dataset.BatchDataset'>['abc'],
    where type('abc') == <class 'str'>
    self:
        ExamplesDataset(len=7)
      SliceDataset(slice(None, 6, None))
    BatchDataset(batch_size=3)

    """
    def __init__(self, input_dataset, batch_size, drop_last=False):
        self.input_dataset = input_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __str__(self):
        return f'{self.__class__.__name__}(batch_size={self.batch_size})'

    def __iter__(self):
        current_batch = list()
        for element in self.input_dataset():
            current_batch.append(element)
            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = list()
        if len(current_batch) > 0 and not self.drop_last:
            yield current_batch

    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            if index < 0:
                # only touch len when necessary
                index = index % len(self)
            input_index = index * self.batch_size
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
            return super().__getitem__(index)

    def __len__(self):
        length = len(self.input_dataset) / self.batch_size
        if self.drop_last:
            return int(length)
        return int(np.ceil(length))


class DynamicBucketDataset(Dataset):
    """
    >>> samples = [1, 10, 5, 7, 8, 2, 4]
    >>> batch_dataset = DynamicBucketDataset(\
    samples, 2, key=lambda x: x, min_rate=0.5)
    >>> [batch for batch in batch_dataset]
    [[10, 5], [7, 8], [1, 2], [4]]
    >>> batch_dataset = DynamicBucketDataset(\
    samples, 2, key=lambda x: x, min_rate=0.5, drop_last=True)
    >>> [batch for batch in batch_dataset]
    [[10, 5], [7, 8], [1, 2]]
    >>> batch_dataset = DynamicBucketDataset(\
    samples, 2, key=lambda x: x, min_rate=0.8)
    >>> [batch for batch in batch_dataset]
    [[10, 8], [5, 4], [1], [7], [2]]
    """
    def __init__(
            self, input_dataset, batch_size, key, min_rate=0.5, max_value=1e6,
            drop_last=False
    ):
        self.input_dataset = input_dataset
        self.batch_size = batch_size
        if callable(key):
            self.key = key
        elif isinstance(key, str):
            self.key = lambda x: x[key]
        else:
            raise ValueError(key)
        self.max_value = max_value
        self.min_rate = min_rate
        self.drop_last = drop_last

    def __iter__(self):
        buckets = list()
        for i, sample in enumerate(self.input_dataset):
            value = min(self.key(sample), self.max_value)
            found_bucket = False
            for i, (bucket, min_value, max_value) in enumerate(buckets):
                if min_value <= value <= max_value:
                    bucket.append(sample)
                    if len(bucket) >= self.batch_size:
                        buckets.pop(i)
                        yield bucket
                    else:
                        min_value = max(min_value, value*self.min_rate)
                        max_value = min(max_value, value/self.min_rate)
                        buckets[i] = (bucket, min_value, max_value)
                    found_bucket = True
                    break
            if not found_bucket:
                buckets.append(([sample], value*self.min_rate, value/self.min_rate))
        if not self.drop_last:
            buckets = sorted(buckets, key=lambda x: len(x[0]), reverse=True)
            for bucket, _, _ in buckets:
                yield bucket


def recursive_transform(func, dict_list_val, list2array=False):
    """
    Applies a function func to all leaf values in a dict or list or directly to
    a value. The hierarchy of dict_list_val is inherited. Lists are stacked
    to numpy arrays. This function can e.g. be used to recursively apply a
    transformation (e.g. audioread) to all audio paths in an example dict
    (see top of this file).
    :param func: a transformation function to be applied to the leaf values
    :param dict_list_val: dict list or value
    :param iterate_lists:
    :param list2array:
    :return: dict, list or value with transformed elements
    """
    if isinstance(dict_list_val, dict):
        # Recursively call itself
        return {key: recursive_transform(func, val, list2array)
                for key, val in dict_list_val.items()}
    if isinstance(dict_list_val, (list, tuple)):
        # Recursively call itself
        l = type(dict_list_val)(
            [recursive_transform(func, val, list2array)
             for val in dict_list_val]
        )
        if list2array:
            return np.array(l)
        return l
    else:
        # applies function to a leaf value
        return func(dict_list_val)


class AudioReader:
    def __init__(self, src_key='audio_path', dst_key='audio_data',
                 audio_keys='observation', read_fn=lambda x: audioread(x)[0],
                 optional_audio_keys=None):
        """
        recursively read audio files and add audio
        signals to the example dict.
        :param src_key: key in an example dict where audio file paths can be
            found.
        :param dst_key: key to add the read audio to the example dict.
        :param audio_keys: str or list of subkeys that are relevant. This can be
            used to prevent unnecessary audioread.
        :param optional_audio_keys: str or list of subkeys to read if present.
        """
        self.src_key = src_key
        self.dst_key = dst_key
        if audio_keys is not None:
            self.audio_keys = to_list(audio_keys)
        else:
            self.audio_keys = None
        if optional_audio_keys is not None:
            self.optional_audio_keys = to_list(optional_audio_keys)
        else:
            self.optional_audio_keys = None
        self._read_fn = read_fn

    def __call__(self, example):
        """
        :param example: example dict with src_key in it
        :return: example dict with audio data added
        """
        if self.audio_keys is not None:
            assert isinstance(example[self.src_key], dict), (
                "example[self.src_key] is not a dict. You probably want to "
                f"set audio_keys to None: {example[self.src_key]}"
            )
            keys = list(example[self.src_key].keys())
            for audio_key in self.audio_keys:
                assert audio_key in keys, (
                    f'Trying to read {audio_key} but only {keys} are available'
                )
            data = {
                audio_key: recursive_transform(
                    self._read_fn, example[self.src_key][audio_key],
                    list2array=True
                )
                for audio_key in self.audio_keys
            }
            if self.optional_audio_keys is not None:
                data.update({
                    audio_key: recursive_transform(
                        self._read_fn, example[self.src_key][audio_key],
                        list2array=True
                    )
                    for audio_key in self.optional_audio_keys
                    if audio_key in example[self.src_key]
                })
        else:
            data = recursive_transform(
                self._read_fn, example[self.src_key], list2array=True
            )

        if self.dst_key is not None:
            example[self.dst_key] = data
        else:
            example.update(data)
        return example


class IdFilter:
    def __init__(self, id_list):
        """
        A filter to filter example ids.
        :param id_list: list of valid ids, e.g. ids belonging to a specific
            dataset.

        An alternative with slicing:

        >>> it = DictDataset({'a': {}, 'b': {}, 'c': {}})
        >>> it = it.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> list(it)
        [{'example_id': 'a'}, {'example_id': 'b'}, {'example_id': 'c'}]
        >>> it['a']
        {'example_id': 'a'}
        >>> it['a', 'b']  # doctest: +ELLIPSIS
              ExamplesDataset(len=3)
              ExamplesDataset(len=3)
            ZipDataset()
          MapDataset(<function <lambda> at 0x...>)
        SliceDataset(('a', 'b'))
        >>> list(it['a', 'b'])
        [{'example_id': 'a'}, {'example_id': 'b'}]

        >>> it.filter(IdFilter(('a', 'b')))  # doctest: +ELLIPSIS
              ExamplesDataset(len=3)
              ExamplesDataset(len=3)
            ZipDataset()
          MapDataset(<function <lambda> at 0x...>)
        FilterDataset(<nt.database.dataset.IdFilter object at 0x...>)
        >>> list(it.filter(IdFilter(('a', 'b'))))
        [{'example_id': 'a'}, {'example_id': 'b'}]
        """
        self.id_list = id_list

    def __call__(self, example):
        """
        :param example: example dict with example_id in it
        :return: True if example_id in id_list else False
        """
        return example[keys.EXAMPLE_ID] in self.id_list


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


class AlignmentReader:
    def __init__(
            self, alignment_path: Path = None, alignments: dict = None,
            example_id_map_fn=lambda x: x[keys.EXAMPLE_ID],
            dst_key=keys.ALIGNMENT):
        assert alignment_path is not None or alignments is not None, (
            'Either alignments or the path to the alignments must be specified'
        )
        self._ali_path = alignment_path
        self._alignments = alignments
        self._map_fn = example_id_map_fn
        self._dst_key = dst_key

    @cached_property
    def alignments(self):
        if self._alignments is None:
            self._alignments =  kaldi.alignment.import_alignment_data(
                self._ali_path
            )
            LOG.debug(
                f'Read {len(self._alignments)} alignments '
                f'from path {self._ali_path}'
            )
        return self._alignments

    def filter_fn(self, example):
        """
        Indicate it there are alignments for an example.

        Usage:
            it = it.filter(alignment_reader.filter_fn, lazy=False)
        """
        value = self._map_fn(example) in self.alignments
        if not value:
            LOG.warning(
                f'No alignment found for example id {example[keys.EXAMPLE_ID]}'
                f' (mapped: {self._map_fn(example)}).'
            )
        return value

    def __call__(self, example):
        try:
            example[self._dst_key] = self.alignments[
                self._map_fn(example)
            ]
            example[keys.NUM_ALIGNMENT_FRAMES] = len(example[self._dst_key])
        except KeyError:
            LOG.warning(
                f'No alignment found for example id {example[keys.EXAMPLE_ID]} '
                f'(mapped: {self._map_fn(example)}).'
            )
        return example


class ExamplesWithoutAlignmentRemover:
    def __init__(self, alignment_key=keys.ALIGNMENT):
        self._key = alignment_key

    def __call__(self, example):
        valid_ali = self._key in example and len(example[self._key])
        if not valid_ali:
            LOG.warning(
                f'Removing example {example[keys.EXAMPLE_ID]} because '
                f'it has no alignment')
            return False
        if keys.NUM_SAMPLES in example:
            num_samples = example[keys.NUM_SAMPLES]
            if isinstance(num_samples, dict):
                num_samples = num_samples[keys.OBSERVATION]
        else:
            return True  # Only happens for Kaldi databases

        # TODO: This assumes fixed size and shift. Does not work for 8 kHz.
        num_frames = (num_samples - 400 + 160) // 160
        num_frames_lfr = (num_frames + np.mod(-num_frames, 3)) // 3
        len_ali = len(example[self._key])
        valid_ali = (
            len_ali == num_frames or
            len_ali == num_frames_lfr
        )
        if not valid_ali:
            LOG.warning(
                f'Example {example[keys.EXAMPLE_ID]}: Alignment has {len_ali} '
                f'frames but the observation has '
                f'{num_frames} [{num_frames_lfr}] frames. Dropping example.'
            )
            return False
        return True


def remove_examples_without_alignment(example):
    return ExamplesWithoutAlignmentRemover()(example)


def remove_zero_length_example(example, audio_key='observation',
                               dst_key='audio_data'):

    if keys.NUM_SAMPLES in example:
        num_samples = example[keys.NUM_SAMPLES]
        if isinstance(num_samples, dict):
            num_samples = num_samples[keys.OBSERVATION]
        valid_ali = num_samples > 0
    else:
        valid_ali = len(example[dst_key][audio_key]) > 0
    if not valid_ali:
        LOG.warning(f'Skipping: Audio length '
                    f'example\n{example[keys.EXAMPLE_ID]} is 0')
        return False
    return True


class LimitAudioLength:
    def __init__(self, max_lengths=160000, audio_keys=('observation',),
                 dst_key='audio_data', frame_length=400, frame_step=160):
        self.max_lengths = max_lengths
        self.audio_keys = audio_keys
        self.dst_key = dst_key
        self.frame_length = frame_length
        self.frame_step = frame_step
        if self.max_lengths:
            LOG.info(f'Will limit audio length to {self.max_lengths}')

    def _sample_to_frame(self, s):
        return max(
            0,
            (s - self.frame_length + self.frame_step) // self.frame_step
        )

    def _frame_to_lfr_frame(self, f):
        return (f + np.mod(-f, 3)) // 3

    def __call__(self, example):
        valid_ex = keys.NUM_SAMPLES in example and \
            example[keys.NUM_SAMPLES] <= self.max_lengths
        orig_len = example[keys.NUM_SAMPLES]
        example['num_dismissed_samples'] = 0
        if not valid_ex:
            delta = max(1, (example[keys.NUM_SAMPLES] - self.max_lengths) // 2)
            start = np.random.choice(delta, 1)[0]

            # audio
            def cut_fn(x): return x[..., start: start + self.max_lengths]
            if self.audio_keys is not None:
                example[keys.AUDIO_DATA] = {
                    audio_key: recursive_transform(
                        cut_fn, example[keys.AUDIO_DATA][audio_key],
                        list2array=True
                    )
                    for audio_key in self.audio_keys
                    if audio_key in example[keys.AUDIO_DATA]
                }
            else:
                example[keys.AUDIO_DATA] = recursive_transform(
                    cut_fn, example[keys.AUDIO_DATA], list2array=True
                )

            # alignment
            if keys.ALIGNMENT in example:
                start_frame = self._sample_to_frame(start)
                new_num_frames = self._sample_to_frame(self.max_lengths)
                # Check for LFR
                num_frames = (example[keys.NUM_SAMPLES] - 400 + 160) // 160
                num_frames_lfr = self._frame_to_lfr_frame(num_frames)
                is_lfr = len(example[keys.ALIGNMENT]) == num_frames_lfr
                if is_lfr:
                    start_frame = self._frame_to_lfr_frame(start_frame)
                    new_num_frames = self._frame_to_lfr_frame(new_num_frames)
                # Adjust alignment
                example[keys.ALIGNMENT] = \
                    example[keys.ALIGNMENT][start_frame: start_frame
                                            + new_num_frames]
                example[keys.NUM_ALIGNMENT_FRAMES] = new_num_frames

            example[keys.NUM_SAMPLES] = self.max_lengths
            example['num_dismissed_samples'] = orig_len - self.max_lengths

            LOG.warning(
                f'Cutting {example[keys.EXAMPLE_ID]}'
                f' to length {self.max_lengths}'
                f' start_frame: {start_frame}'
                f' new_num_frames: {new_num_frames}'
                f' lfr: {is_lfr}'
            )
        return example


class Word2Id:
    def __init__(self, word2id_fn):
        def _safe_word_to_id_fn(w):
            try:
                return word2id_fn(w)
            except KeyError:
                LOG.warning(f'Could not resolve ID for word {w}')
                return

        self._word2id_fn = _safe_word_to_id_fn

    def __call__(self, example):
        def _w2id(s):
            return np.array(
                [self._word2id_fn(w) for w in s.split() if self._word2id_fn(w)],
                np.int32)

        if not (keys.TRANSCRIPTION in example or
                keys.KALDI_TRANSCRIPTION in example):
            raise ValueError(
                'Could not find transcription for example id '
                f'{example[keys.EXAMPLE_ID]}'
            )
        if keys.TRANSCRIPTION in example:
            example[keys.TRANSCRIPTION + '_ids'] = recursive_transform(
                _w2id, example[keys.TRANSCRIPTION]
            )
        if keys.KALDI_TRANSCRIPTION in example:
            example[keys.KALDI_TRANSCRIPTION + '_ids'] = recursive_transform(
                _w2id, example[keys.KALDI_TRANSCRIPTION]
            )

        return example