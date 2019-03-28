import pytest

import lazy_dataset


def get_examples():
    examples = {
        'example_id_1': {
            'observation': [1, 2, 3],
            'label': 1,
        },
        'example_id_2': {
            'observation': [4, 5, 6],
            'label': 2,
        },
        'example_id_3': {
            'observation': [7, 8, 9],
            'label': 3,
        },
    }
    for example_id, example in examples.items():
        example['example_id'] = example_id
    return examples


def get_examples_predict():
    examples = {
        'example_id_4': {
            'observation': [10, 11, 12],
        },
        'example_id_5': {
            'observation': [13, 14, 15],
        },
    }
    for example_id, example in examples.items():
        example['example_id'] = example_id
    return examples


def get_dataset():
    examples = get_examples()
    return lazy_dataset.new(examples)


def get_dataset_predict():
    examples = get_examples_predict()
    return lazy_dataset.new(examples)


def test_keys():
    examples = get_examples()
    ds = get_dataset()

    assert ds.keys() == tuple(examples.keys())


def test_getitem(ds=None):
    if ds is None:
        ds = get_dataset()
    ex1_str = ds['example_id_1']
    ex2_str = ds['example_id_2']
    ex1_int = ds[0]
    ex2_int = ds[1]
    ex1_slice_int = ds[:1][0]

    assert ex1_str == ex1_int
    assert ex1_str == ex1_slice_int
    assert ex2_str == ex2_int


def test_contains():
    ds = get_dataset()

    with pytest.raises(Exception):
        # contains should be unsupported
        'example_id_1' in ds


def test_map():
    ds = get_dataset()

    def map_fn(d):
        d['example_id'] = d['example_id'].upper()
        return d

    iterator = ds.map(map_fn)
    example_ids = [e['example_id'] for e in iterator]
    assert example_ids == 'EXAMPLE_ID_1 EXAMPLE_ID_2 EXAMPLE_ID_3'.split()

    # Getitem should still be supported
    test_getitem(ds)


def test_filter():
    ds = get_dataset()

    def filter_fn(d):
        return not d['example_id'] == 'example_id_2'

    iterator = ds.filter(filter_fn)
    example_ids = [e['example_id'] for e in iterator]
    assert example_ids == 'example_id_1 example_id_3'.split()

    # Getitem with str should be supported
    _ = ds['example_id_1']

    # Getitem with filtered str should fail
    with pytest.raises(IndexError):
        _ = iterator['example_id_2']

    # Getitem with int is not supported
    with pytest.raises(AssertionError):
        _ = iterator[0]
    with pytest.raises(AssertionError):
        _ = iterator[:1]


def test_concatenate_function():
    ds_train = get_dataset()
    ds_predict = get_dataset_predict()

    ds = lazy_dataset.concatenate(ds_train, ds_predict)
    example_ids = [e['example_id'] for e in ds]
    assert example_ids == [f'example_id_{i}' for i in range(1, 6)]

    assert ds['example_id_1']['example_id'] == 'example_id_1'
    assert ds['example_id_5']['example_id'] == 'example_id_5'
    assert ds[0]['example_id'] == 'example_id_1'
    assert ds[-1]['example_id'] == 'example_id_5'
    assert ds[:1][0]['example_id'] == 'example_id_1'


def test_concatenate_function_raises_on_empty_list():
    with pytest.raises(ValueError):
        lazy_dataset.concatenate()


def test_concatenate_function_raises_on_non_dataset_instances():
    ds_train = get_dataset()
    not_a_ds = dict()
    with pytest.raises(TypeError):
        lazy_dataset.concatenate(ds_train, not_a_ds)


def test_concatenate():
    ds_train = get_dataset()
    ds_predict = get_dataset_predict()

    ds = ds_train.concatenate(ds_predict)
    example_ids = [e['example_id'] for e in ds]
    assert example_ids == [f'example_id_{i}' for i in range(1, 6)]

    assert ds['example_id_1']['example_id'] == 'example_id_1'
    assert ds['example_id_5']['example_id'] == 'example_id_5'
    assert ds[0]['example_id'] == 'example_id_1'
    assert ds[-1]['example_id'] == 'example_id_5'
    assert ds[:1][0]['example_id'] == 'example_id_1'


def test_concatenate_double_keys():
    ds_1 = get_dataset()
    ds_2 = get_dataset()
    ds = ds_1.concatenate(ds_2)
    example_ids = [e['example_id'] for e in ds]
    assert example_ids == ['example_id_1', 'example_id_2', 'example_id_3', 'example_id_1', 'example_id_2', 'example_id_3']

    with pytest.raises(AssertionError):
        _ = ds['a']

    assert ds[0]['example_id'] == 'example_id_1'
    assert ds[-1]['example_id'] == 'example_id_3'
    assert ds[:1][0]['example_id'] == 'example_id_1'


def test_zip():
    import numpy as np
    ds = get_dataset()

    # Change the key order
    np.random.seed(2)
    ds_shuffled = ds.shuffle(reshuffle=False)

    example_ids = [e['example_id'] for e in ds]
    assert example_ids == ['example_id_1', 'example_id_2', 'example_id_3']
    example_ids = [e['example_id'] for e in ds_shuffled]
    assert example_ids == ['example_id_3', 'example_id_2', 'example_id_1']

    ds_zip = ds.zip(ds_shuffled)
    ds_shuffled_zip = ds_shuffled.zip(ds)

    assert ds_zip[0] == (
        {'observation': [1, 2, 3], 'label': 1, 'example_id': 'example_id_1'},
        {'observation': [1, 2, 3], 'label': 1, 'example_id': 'example_id_1'},
    )
    assert ds_shuffled_zip[0] == (
        {'observation': [7, 8, 9], 'label': 3, 'example_id': 'example_id_3'},
        {'observation': [7, 8, 9], 'label': 3, 'example_id': 'example_id_3'},
    )


def test_slice():
    ds = get_dataset()
    ds = ds.concatenate(ds)

    ds_slice = ds[:6]
    example_ids = [e['example_id'] for e in ds_slice]
    assert example_ids == ['example_id_1', 'example_id_2', 'example_id_3', 'example_id_1', 'example_id_2', 'example_id_3']

    ds_slice = ds[:4]
    example_ids = [e['example_id'] for e in ds_slice]
    assert example_ids == ['example_id_1', 'example_id_2', 'example_id_3', 'example_id_1']

    ds_slice = ds[:20]  # Should this work? Work for list.
    example_ids = [e['example_id'] for e in ds_slice]
    assert example_ids == ['example_id_1', 'example_id_2', 'example_id_3', 'example_id_1', 'example_id_2', 'example_id_3']

    _ = ds[:2]
    _ = ds[:1]
    _ = ds[:0]  # Should this work? Work for list.
