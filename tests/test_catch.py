import pytest

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


def test_catch():
    ds = get_dataset()

    def map_function(example):
        if example['value'] == 3:
            raise lazy_dataset.FilterException('Got 3')
        return example

    ds = ds.map(map_function)

    with pytest.raises(lazy_dataset.FilterException):
        list(ds)

    ds = ds.catch()

    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b d e'.split()


def test_catch_with_shuffle():
    ds = get_dataset().shuffle(reshuffle=True)

    def map_function(example):
        if example['value'] == 3:
            raise lazy_dataset.FilterException('Got 3')
        return example

    ds = ds.map(map_function)

    with pytest.raises(lazy_dataset.FilterException):
        list(ds)

    ds = ds.catch()

    example_ids = [ex['example_id'] for ex in ds]
    assert set(example_ids) == set('a b d e'.split())
