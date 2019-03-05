import pytest
import numpy as np

import lazy_dataset


def get_examples():
    examples = {
        'a': {},
        'b': {},
        'c': {},
        'd': {},
        'e': {},
    }
    for example_id, example in examples.items():
        example['example_id'] = example_id
    return examples


def get_dataset():
    examples = get_examples()
    return lazy_dataset.new(examples)


def test_shuffle_prefetch():
    rng = np.random.RandomState(1)

    # Without reshuffle prefetch works fine, but the example order stays the
    # same
    ds = get_dataset()
    ds = ds.shuffle(rng=rng, reshuffle=False)
    ds = ds.prefetch(2, 4)

    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'c b e a d'.split()
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'c b e a d'.split()

    np.random.seed(0)

    # With reshuffle prefetch does not work
    ds = get_dataset()
    ds = ds.shuffle(reshuffle=True)
    ds = ds.prefetch(2, 4)
    with pytest.raises(TypeError):
        example_ids = [ex['example_id'] for ex in ds]

    np.random.seed(0)

    # With reshuffle and freeze in prefetch, shuffling works and the example
    # order changes each time
    ds = get_dataset()
    ds = ds.shuffle(reshuffle=True)
    ds = ds.prefetch(2, 4, freeze=True)
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'c a b d e'.split()
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'a c b e d'.split()
