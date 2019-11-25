import lazy_dataset


def test_key_zip():
    ds1 = lazy_dataset.new({'1': 1, '2': 2, '3': 3, '4': 4})
    ds2 = lazy_dataset.new({'1': 5, '2': 6, '3': 7, '4': 8})
    ds = lazy_dataset.key_zip(ds1, ds2)
    assert list(ds) == [(1, 5), (2, 6), (3, 7), (4, 8)]
    ds = lazy_dataset.key_zip(ds1, ds2).prefetch(2, 2)
    assert list(ds) == [(1, 5), (2, 6), (3, 7), (4, 8)]


def test_zip():
    ds1 = lazy_dataset.new({'1_1': 1, '1_2': 2, '1_3': 3, '1_4': 4})
    ds2 = lazy_dataset.new({'2_1': 5, '2_2': 6, '2_3': 7, '2_4': 8})
    ds = lazy_dataset.zip(ds1, ds2)
    assert list(ds) == [(1, 5), (2, 6), (3, 7), (4, 8)]
    ds = lazy_dataset.zip(ds1, ds2).prefetch(2, 2)
    assert list(ds) == [(1, 5), (2, 6), (3, 7), (4, 8)]
