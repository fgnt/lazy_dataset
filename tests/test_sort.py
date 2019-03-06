import pytest

import lazy_dataset


def get_examples():
    examples = {
        'b': {'value': 2},
        'c': {'value': 3},
        'a': {'value': 1},
        'e': {'value': 5},
        'd': {'value': 4},
    }
    for example_id, example in examples.items():
        example['example_id'] = example_id
    return examples


def get_dataset():
    examples = get_examples()
    return lazy_dataset.new(examples)


def test_sort():

    def sort_fn(example):
        return example['value']

    ds = get_dataset()
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'b c a e d'.split()
    ds = ds.sort(sort_fn)
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a b c d e'.split()


def test_sort_reverse():

    def sort_fn(example):
        return example['value']

    ds = get_dataset()
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'b c a e d'.split()
    ds = ds.sort(sort_fn, reverse=True)
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'e d c b a'.split()
