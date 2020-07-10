import tempfile
from collections import Counter
from pathlib import Path

import lazy_dataset
import psutil
import mock
import pytest


def gb(x):
    return x * 1024 ** 3


def test_cache_immutable():
    dataset = lazy_dataset.new({
        'a': {'value': 1},
        'b': {'value': 2},
        'c': {'value': 3},
    })

    cached_dataset = dataset.cache()

    assert cached_dataset['a']['value'] == 1
    cached_dataset['a']['value'] = 42
    assert cached_dataset['a']['value'] == 1


def test_cache_call_only_once():
    call_counter = Counter()
    dataset = lazy_dataset.new(dict(zip(map(str, range(10)), range(10))))

    def m(x):
        call_counter[x] += 1
        return x

    # Set keep_mem_free to a small value to allow testing on machines with
    # less RAM
    dataset = dataset.map(m).cache(keep_mem_free='1GB')
    for _ in dataset:
        pass
    assert all(v == 1 for v in call_counter.values())
    for _ in dataset:
        pass
    assert all(v == 1 for v in call_counter.values())


def test_cache_mem_abs():
    dataset = lazy_dataset.new(dict(zip(map(str, range(100)), range(100))))

    available_mem = gb(8)
    def virtual_memory():
        return psutil._pslinux.svmem(
            total=gb(16), available=available_mem, percent=0, used=0, free=0,
            active=0, inactive=0, buffers=0, cached=0, shared=0, slab=0
        )

    with mock.patch('psutil.virtual_memory', new=virtual_memory):
        # Test absolute
        ds = dataset.cache(keep_mem_free='5GB')

        available_mem = gb(6)
        it = iter(ds)
        assert len(ds._cache) == 0
        next(it)
        assert len(ds._cache) == 1
        available_mem = gb(5)
        with pytest.warns(ResourceWarning, match='Max capacity'):
            next(it)
        assert len(ds._cache) == 1


def test_cache_mem_percent():
    dataset = lazy_dataset.new(dict(zip(map(str, range(100)), range(100))))

    available_mem = gb(8)
    def virtual_memory():
        return psutil._pslinux.svmem(
            total=gb(16), available=available_mem, percent=0, used=0, free=0,
            active=0, inactive=0, buffers=0, cached=0, shared=0, slab=0
        )

    with mock.patch('psutil.virtual_memory', new=virtual_memory):
        ds = dataset.cache(keep_mem_free='50%')

        available_mem = gb(9)
        it = iter(ds)
        assert len(ds._cache) == 0
        next(it)
        assert len(ds._cache) == 1
        available_mem = gb(7)
        with pytest.warns(ResourceWarning, match='Max capacity'):
            next(it)
        assert len(ds._cache) == 1


def test_cache_from_ordered_not_indexable():
    dataset = lazy_dataset.new(list(range(10)))
    assert len(dataset.filter(lambda x: x % 2).cache(lazy=False)) == 5


def test_cache_no_keys():
    ds = lazy_dataset.new(list(range(100))).cache()
    assert list(ds) == list(range(100))


def test_diskcache_immutable():
    dataset = lazy_dataset.new({
        'a': {'value': 1},
        'b': {'value': 2},
        'c': {'value': 3},
    })

    cached_dataset = dataset.diskcache()

    assert cached_dataset['a']['value'] == 1
    cached_dataset['a']['value'] = 42
    assert cached_dataset['a']['value'] == 1


def test_diskcache_call_only_once():
    call_counter = Counter()
    dataset = lazy_dataset.new(dict(zip(map(str, range(10)), range(10))))

    def m(x):
        call_counter[x] += 1
        return x

    dataset = dataset.map(m).diskcache()
    for _ in dataset:
        pass
    assert all(v == 1 for v in call_counter.values())
    for _ in dataset:
        pass
    assert all(v == 1 for v in call_counter.values())


def test_diskcache_clear():
    dataset = lazy_dataset.new(list(range(10)))
    cache_dir = tempfile.mkdtemp()

    dataset = dataset.diskcache(cache_dir=cache_dir, clear=True)
    list(dataset)
    assert Path(cache_dir).is_dir()
    del(dataset)
    assert not Path(cache_dir).exists()


def test_diskcache_reuse():
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create dataset and write to cache
        dataset = lazy_dataset.new(list(range(10))).diskcache(
            cache_dir=cache_dir, reuse=True, clear=False)
        list(dataset)
        del dataset

        # Assert that the cache didn't get deleted
        assert Path(cache_dir).exists()

        # Create a new dataset with the same cache dir and assert that the data
        # gets loaded from the cache and not from the data pipeline
        call_counter = 0

        def _count_calls(x):
            nonlocal call_counter
            call_counter += 1
            return x

        dataset = (lazy_dataset.new(list(range(10)))
            .map(_count_calls)
            .diskcache(cache_dir=cache_dir, reuse=True, clear=False))
        list(dataset)

        assert call_counter == 0


def test_diskcache_raise_if_cache_exists():
    with tempfile.TemporaryDirectory() as cache_dir:
        (Path(cache_dir) / 'file').touch()
        with pytest.raises(RuntimeError):
            lazy_dataset.new(list(range(10))).diskcache(
                cache_dir=cache_dir, reuse=False)
