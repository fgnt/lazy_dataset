from collections import Counter

import lazy_dataset
import numpy as np
import psutil
import mock


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

    dataset = dataset.map(m).cache()
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
        assert len(ds.cache) == 0
        next(it)
        assert len(ds.cache) == 1
        available_mem = gb(5)
        next(it)
        assert len(ds.cache) == 1


def test_cache_mem_fraction():
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
        assert len(ds.cache) == 0
        next(it)
        assert len(ds.cache) == 1
        available_mem = gb(7)
        next(it)
        assert len(ds.cache) == 1
