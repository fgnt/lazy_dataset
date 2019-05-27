import lazy_dataset
import numpy as np


def test_bucket_idx():
    examples = list(range(10))
    examples = {str(j): i for j, i in enumerate(examples)}
    ds = lazy_dataset.new(examples)
    idx = list(ds.get_bucket_indices(lambda x: x, 5).values())
    assert (np.array(idx) == np.repeat(list(range(5)), 2)).all()


def test_bucket():
    examples = [1, 10, 5, 7, 8, 2, 4, 3, 20, 1, 6, 9]
    examples = {str(j): i for j, i in enumerate(examples)}
    ds = lazy_dataset.new(examples)

    idx = ds.get_bucket_indices(lambda x: x, 3)
    buckets = [list(bucket) for bucket in ds.bucket(idx)]
    assert buckets == [
        [1, 2, 3, 1],
        [5, 7, 4, 6],
        [10, 8, 20, 9]
    ]

    batched_buckets = list(ds.batch_bucket(bucket_indices=idx, batch_size=2))
    assert batched_buckets == [[1, 2], [3, 1], [5, 7], [4, 6], [10, 8], [20, 9]]

    dynamic_batched_buckets = list(ds.batch_bucket_dynamic(
        batch_size=2, key=lambda x: x, max_padding_rate=0.5
    ))
    assert dynamic_batched_buckets == [
        [10, 5], [7, 8], [1, 2], [4, 3], [6, 9], [20], [1]
    ]
