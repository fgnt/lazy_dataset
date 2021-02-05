import lazy_dataset


def assert_doctest_equal(got, want, options=('ELLIPSIS',)):
    import doctest
    assert isinstance(got, str), got

    optionflags = 0
    for o in options:
        optionflags |= doctest.OPTIONFLAGS_BY_NAME[o]

    checker = doctest.OutputChecker()
    checked = checker.check_output(want, got, optionflags)
    if not checked:
        raise AssertionError(checker.output_difference(
            doctest.Example('dummy', want),
            got + '\n',
            optionflags,
        ).rstrip('\n'))


def test():
    def check(ds, expected_str, expected_repr=None):
        assert_doctest_equal(str(ds), expected_str)
        if expected_repr is None:
            repr(ds_dict)
        else:
            assert_doctest_equal(repr(ds), expected_repr)

    ds_dict = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3, 'd': 4}, immutable_warranty='copy')
    check(ds_dict,
          'MapDataset(copy.deepcopy)',
          '  DictDataset(len=4)\n'
          'MapDataset(copy.deepcopy)')

    ds_dict = lazy_dataset.new({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    check(ds_dict,
          'MapDataset(_pickle.loads)',
          '  DictDataset(len=4)\n'
          'MapDataset(_pickle.loads)')

    ds_list = lazy_dataset.new([1, 2, 3, 4])
    check(ds_list,
          'MapDataset(_pickle.loads)',
          '  ListDataset(len=4)\n'
          'MapDataset(_pickle.loads)')

    ds = ds_dict.map(lambda ex: ex)
    check(ds, 'MapDataset(<function test.<locals>.<lambda> at 0x...>)')

    ds = ds_dict.filter(lambda ex: True)
    check(ds, 'FilterDataset(<function test.<locals>.<lambda> at 0x...>)')

    ds = ds_dict.cache(keep_mem_free='5 GiB')
    check(ds,
          'CacheDataset(keep_free=5 GiB)',
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          'CacheDataset(keep_free=5 GiB)')
    ds = ds_dict.cache(keep_mem_free='5 GiB').copy()
    check(ds,
          'CacheDataset(keep_free=5 GiB)',
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          'CacheDataset(keep_free=5 GiB)')

    ds = ds_dict.filter(lambda ex: True).cache(lazy=False)
    check(ds, 'MapDataset(_pickle.loads)',
          '  ListDataset(len=4)\n'
          'MapDataset(_pickle.loads)')

    ds = ds_dict.diskcache()
    check(ds,
          'DiskCacheDataset(cache_dir=/tmp/diskcache-..., reuse=False)')

    ds = ds_dict.diskcache().copy()
    check(ds,
          'DiskCacheDataset(cache_dir=/tmp/diskcache-..., reuse=False)')

    ds = ds_dict.random_choice(2)
    check(ds, 'SliceDataset([...])')

    ds = ds_dict.tile(2)
    check(ds, 'ConcatenateDataset()',
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          'ConcatenateDataset()')

    ds = ds_dict.batch(2)
    check(ds, 'BatchDataset(batch_size=2)')

    ds = ds.unbatch()
    check(ds, 'UnbatchDataset()')

    ds = ds_dict.catch()
    check(ds, 'CatchExceptionDataset()')

    ds = ds_dict.shuffle(reshuffle=True)
    check(ds, 'ReShuffleDataset()')

    ds = ds_dict.shuffle(reshuffle=False)
    check(ds, 'SliceDataset([...])')

    ds = ds_dict.sort()  # sort by keys
    check(ds, "SliceDataset(['a', 'b', 'c', 'd'])")

    ds = ds_list.sort(key_fn=lambda ex: ex)
    check(ds, 'SliceDataset([0, 1, 2, 3])')

    ds = ds_dict.zip(ds_list)
    check(ds, 'ZipDataset()',
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          '    ListDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          'ZipDataset()')

    import numpy as np
    np.random.seed(0)
    ds = ds_dict.key_zip(ds_dict.shuffle())
    check(ds, 'KeyZipDataset()',
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          '      DictDataset(len=4)\n'
          '    MapDataset(_pickle.loads)\n'
          '  SliceDataset([2 3 1 0])\n'
          'KeyZipDataset()')

    ds = ds_dict.intersperse(ds_list)
    check(ds, 'IntersperseDataset()',
          '    DictDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          '    ListDataset(len=4)\n'
          '  MapDataset(_pickle.loads)\n'
          'IntersperseDataset()')
