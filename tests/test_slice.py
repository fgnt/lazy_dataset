import pytest
import numpy as np

import lazy_dataset


def get_examples():
    examples = {
        'a': {'value': 1},
        'b': {'value': 2},
        'c': {'value': 3},
        'd': {'value': 4},
        'e': {'value': 5},
    }
    for example_id, example in examples.items():
        example['example_id'] = example_id
    return examples


def get_dataset():
    examples = get_examples()
    return lazy_dataset.new(examples)


def test_slice():
    ds = get_dataset()[:2]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_strs():
    ds = get_dataset()['a', 'b']
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_tuple_str():
    ds = get_dataset()[('a', 'b')]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_list_str():
    ds = get_dataset()[['a', 'b']]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_ints():
    with pytest.raises(IndexError):
        _ = np.arange(5)[0, 1]

    ds = get_dataset()[0, 1]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_tuple_int():
    with pytest.raises(IndexError):
        _ = np.arange(5)[(0, 1)]

    ds = get_dataset()[(0, 1)]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_list_int():
    _ = np.arange(5)[[0, 1]]

    ds = get_dataset()[[0, 1]]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_tuple_int_komma():
    _ = np.arange(5)[(0, 1),]

    ds = get_dataset()[(0, 1),]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()


def test_list_int_komma():
    _ = np.arange(5)[[0, 1],]

    ds = get_dataset()[[0, 1],]
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b'.split()
