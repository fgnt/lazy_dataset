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

    rng = np.random.RandomState(1)
    # With reshuffle prefetch still works and the ordering is each time
    # different
    ds = get_dataset()
    ds = ds.shuffle(rng=rng, reshuffle=True)
    ds = ds.prefetch(2, 4)
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'c b e a d'.split()
    example_ids = [ex['example_id'] for ex in ds]
    assert example_ids == 'c e d a b'.split()


def test_reshuffle_slice():
    rng = np.random.RandomState(1)

    ds = get_dataset()
    ds = ds.shuffle(rng=rng, reshuffle=True)
    ds = ds.map(lambda x: x)

    with pytest.raises(RuntimeError):
        ds = ds[:3]
