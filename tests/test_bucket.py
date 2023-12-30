import lazy_dataset


def test_bucket():
    examples = [1, 10, 5, 7, 8, 2, 4, 3, 20, 1, 6, 9]
    examples = {str(j): i for j, i in enumerate(examples)}
    ds = lazy_dataset.new(examples)

    dynamic_batched_buckets = list(ds.batch_dynamic_time_series_bucket(
        batch_size=1, len_key=lambda x: x, max_padding_rate=0.5
    ))
    assert dynamic_batched_buckets == [
        [1], [10], [5], [7], [8], [2], [4], [3], [20], [1], [6], [9]
    ]

    dynamic_batched_buckets = list(ds.batch_dynamic_time_series_bucket(
        batch_size=2, len_key=lambda x: x, max_padding_rate=0.5
    ))
    assert dynamic_batched_buckets == [
        [10, 5], [7, 8], [1, 2], [4, 3], [6, 9], [20], [1]
    ]


def test_max_total_size():
    examples = [6, 7, 9, 5, 6, 3, 7, 4]
    examples = {str(j): i for j, i in enumerate(examples)}
    ds = lazy_dataset.new(examples)

    dynamic_batched_buckets = list(ds.batch_dynamic_time_series_bucket(
        batch_size=3, len_key=lambda x: x, max_padding_rate=0.9, max_total_size=21,
    ))
    assert dynamic_batched_buckets == [
        [6, 7, 5], [9, 6], [3, 7, 4]
    ]
