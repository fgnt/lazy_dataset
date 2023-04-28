import functools
import operator
from pathlib import Path
import pytest
import lazy_dataset
import time


@pytest.mark.parametrize("backend,func", [
    ['t', 'global'],
    ['t', 'local'],
    ['t', 'partial'],
    ['t', 'lambda'],
    ['mp', 'global'],
    ['mp', 'local'],
    # ['mp', 'partial'],  # AttributeError: 'functools.partial' object has no attribute '__module__'
    ['mp', 'lambda'],
    ['dill_mp', 'global'],
    ['dill_mp', 'local'],
    # ['dill_mp', 'partial'],  # AttributeError: 'functools.partial' object has no attribute '__module__'
    ['dill_mp', 'lambda'],
    ['multiprocessing', 'global'],
    # ['multiprocessing', 'local'],  # AttributeError: Can't pickle local object 'test_prefetch.<locals>.foo'
    ['multiprocessing', 'partial'],
    # ['multiprocessing', 'lambda'],  # AttributeError: Can't pickle local object 'test_prefetch.<locals>.<lambda>'
    ['concurrent_mp', 'global'],
    # ['concurrent_mp', 'local'],  # ToDO: Fix deadlock to exception
    ['concurrent_mp', 'partial'],
    # ['concurrent_mp', 'lambda'],  # ToDO: Fix deadlock to exception
])
def test_prefetch(backend, func):
    if func == 'local':
        def foo(x):
            return x + 2
    elif func == 'global':
        foo = operator.neg
    elif func == 'partial':
        foo = functools.partial(operator.add, 2)
    elif func == 'lambda':
        foo = lambda x: x+2
    else:
        raise ValueError(func)

    ds = lazy_dataset.from_list(list(range(100)))
    ds = ds.map(foo)

    ds = ds.prefetch(2, 50, backend=backend)

    for ex in ds:
        pass


def test_break_threads():
    # This test tests a bug from the past:
    # lazy_parallel_map finished all calculations in the buffer, when
    # it received a GeneratorExit.

    data = []
    def foo(ex):
        # Some delay to prevent an immediate full buffer.
        # time.sleep releases the GIL.
        time.sleep(0.01)
        data.append(ex)
        return ex

    ds = lazy_dataset.from_list(list(range(100)))
    ds = ds.map(foo)

    ds = ds.prefetch(2, 50, backend='t')

    for ex in ds:
        break

    assert 1 <= len(data) <= 5, (len(data), data)


class Foo:
    def __init__(self, tmpdir, sleep):
        self.tmpdir = tmpdir
        self.sleep = sleep

    def __call__(self, ex):
        # Some delay to prevent an immediate full buffer.
        # time.sleep releases the GIL.
        time.sleep(self.sleep)
        (self.tmpdir / f'{ex}.txt').touch()
        return ex


@pytest.mark.parametrize("backend,sleep,thresh", [
    ['t', 0.01, 5],
    ['mp', 0.01, 15],
    ['dill_mp', 0.01, 12],
    ['multiprocessing', 0.01, 5],
    ['concurrent_mp', 0.01, 8],
])
def test_break_backend(backend, sleep, thresh):
    # This test tests a bug from the past:
    # lazy_parallel_map finished all calculations in the buffer, when
    # it received a GeneratorExit.

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        ds = lazy_dataset.from_list(list(range(100)))
        ds = ds.map(Foo(tmpdir, sleep))

        # Note: This test assumes a thread backend and also works only with
        # threads.
        ds = ds.prefetch(2, 50, backend=backend)

        for ex in ds:
            break
        time.sleep(1)
        data = list(tmpdir.glob('*'))

        assert 1 <= len(data) <= thresh, (backend, len(data), data)
