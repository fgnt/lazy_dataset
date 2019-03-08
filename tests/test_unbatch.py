import lazy_dataset
from collections import OrderedDict


def test_unbatch():
    examples = OrderedDict(
        a=[0, 1, 2],
        b=[3, 4],
        c=[5, 6, 7]
    )
    ds = lazy_dataset.new(examples)
    ds = ds.unbatch()
    assert list(ds) == list(range(8))


def fragment_fn(ex):
    for i in ex.split('_'):
        yield int(i)


def test_fragment():
    examples = OrderedDict(
        a='0_1_2',
        b='3_4',
        c='5_6_7'
    )
    ds = lazy_dataset.new(examples)
    ds = ds.map(fragment_fn).unbatch()
    assert list(ds) == list(range(8))


def test_prefetch():
    examples = OrderedDict(
        a='0_1_2',
        b='3_4',
        c='5_6_7'
    )
    ds = lazy_dataset.new(examples)
    ds = ds.map(fragment_fn).prefetch(2, 2).unbatch()
    assert list(ds) == list(range(8))
