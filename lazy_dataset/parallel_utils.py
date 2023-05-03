import queue
import concurrent.futures
import os
import sys
import contextlib
import threading
from typing import Iterable, Optional


def ensure_single_thread_numeric():
    """
    When you parallelize your input pipeline you often want each worker to work
    on a single thread.

    These are all candidates to set to 1, but the ones checked in this
    function are mandatory as far as we know.

    GOMP_NUM_THREADS
    OMP_NUM_THREADS
    OPENBLAS_NUM_THREADS
    MKL_NUM_THREADS
    VECLIB_MAXIMUM_THREADS
    NUMEXPR_NUM_THREADS

    Returns:

    """
    candidates = 'OMP_NUM_THREADS MKL_NUM_THREADS'.split()

    for key in candidates:
        if not os.environ.get(key) == '1':
            raise EnvironmentError(
                'Make sure to set the following environment variables to '
                'ensure that each worker works on a single thread:\n'
                'export OMP_NUM_THREADS=1\n'
                'export MKL_NUM_THREADS=1\n\n'
                f'But you use: {key}={os.environ.get(key)}'
            )


def _dill_mp_helper(payload):
    import dill

    fun, args, kwargs = dill.loads(payload)
    return fun(*args, **kwargs)


def lazy_parallel_map(
        function: callable,
        generator: Iterable,
        *,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        backend: str = "t",
        buffer_size: int = 5,
        max_workers: int = 2
):
    """
    This is a parallel map where the function is parallel executed and the
    output is buffered. Note the generator is executed serially.

    A serial version of this function is::

        for ele in generator:
            yield function(ele, *args, **kwargs)

    The function is executed in parallel (not the generator) and the output of
    the function is buffered.

    Available backends are:
        - 'mp': Multiprocessing backend. Uses a
            `pathos.multiprocessing.ProcessPool` to parallelize
        - 'dill_mp': Uses `concurrent.futures.ProcessPoolExecutor` to
            parallelize and `dill` for serialization of arguments and
            results of `function`
        - 't' / 'threaded': Uses a `concurrent.futures.ThreadPoolExecutor`
        - 'concurrent_mp': Uses a `concurrent.futures.ProcessPoolExecutor`.
            Note that this uses `pickle` for serialization, so it can not
            handle arbitrary callable objects as `function`.

    Args:
        function: Function to map. Must accept an element of `generator` as
            first positional argument. The provided `args` and `kwargs` are
            forwarded to `function` as `function(element, *args, **kwargs)`.
        generator: Generator to iterate over
        args: Additional arguments for `function`
        kwargs: Additional keyword arguments for `function`
        backend: The backend to use
        buffer_size: Number of examples to buffer
        max_workers: Maximum number of threads/processes used for parallel
            execution

    Yields:
        The results of function applied to each element of `generator`. The
        order of the elements produces by `generator` is preserved.

    Note:
     - The overhead to copy the data from and to the workers can destroy the
       gain from mutiprocessing ('mp', 'dill_mp').
       Only the threaded backend ('t') has no copy overhead.
     - When the function spends a high amount of time in numpy and/or I/O the
       threaded backend ('t') is recommended. The reason for numpy is, that it
       usually releases the GIL.
     - Do not forget to disable low level parallel execution
       (see `ensure_single_thread_numeric`) when you have CPU bound code.
       For bad combinations, the parallel execution can be slower than the
       serial execution.

    """
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = []

    if max_workers > 1 or backend is False:
        ensure_single_thread_numeric()

    q = queue.Queue()

    if backend is not False:
        assert buffer_size >= max_workers
    assert buffer_size > 0

    if backend == "mp":
        # http://stackoverflow.com/a/21345423
        from pathos.multiprocessing import ProcessPool as PathosPool
        import pathos.helpers.pp_helper
        PoolExecutor = PathosPool

        def submit(ex, func, *args, **kwargs):
            return ex.apipe(func, *args, **kwargs)

        def result(job: pathos.helpers.pp_helper.ApplyResult):
            return job.get()

        def terminate(ex: pathos.multiprocessing.ProcessPool, q):
            ex.terminate()
            # Cancel doesn't work for pathos. Don't know why.
            # try:
            #     while True:
            #         q.get(block=False).cancel()
            # except queue.Empty:
            #     pass

    elif backend == 'multiprocessing':
        from multiprocessing import Pool as PoolExecutor

        def submit(ex, func, *args, **kwargs):
            return ex.apply_async(func, args, kwargs)

        def result(job):
            return job.get()

        def terminate(ex, q):
            # From https://docs.python.org/3/library/multiprocessing.html
            #
            # > multiprocessing.pool.Pool.terminate: Stops the worker processes
            # > immediately without completing outstanding work. When the pool
            # > object is garbage collected terminate() will be called
            # > immediately.
            #
            # Multiprocessing works fine with GeneratorExit and no "hack" is
            # necessary to fix it.
            pass

    elif backend == "dill_mp":
        import dill

        # https://stackoverflow.com/a/24673524
        PoolExecutor = concurrent.futures.ProcessPoolExecutor

        def submit(ex, func, *args, **kwargs):
            payload = dill.dumps((func, args, kwargs))
            return ex.submit(_dill_mp_helper, payload)

        def result(job):
            return job.result()

        def terminate(ex: concurrent.futures.Executor, q):
            try:
                while True:
                    q.get(block=False).cancel()
            except queue.Empty:
                pass

    elif backend in [
            "t",
            "thread",
            "concurrent_mp",
    ]:
        if backend in ['t', 'thread']:
            PoolExecutor = concurrent.futures.ThreadPoolExecutor
        elif backend == "concurrent_mp":
            # does not allow to pickle arbitrary functions
            PoolExecutor = concurrent.futures.ProcessPoolExecutor
        else:
            raise ValueError(backend)

        def submit(ex, func, *args, **kwargs):
            return ex.submit(func, *args, **kwargs)

        def result(job: concurrent.futures.Future):
            return job.result()

        def terminate(ex: concurrent.futures.Executor, q):
            # From https://docs.python.org/3/library/concurrent.futures.html
            #
            # > Regardless of the value of wait, the entire Python program
            # > will not exit until all pending futures are done executing.
            #
            # This is an undesired behaviour for lazy_dataset, i.e., when
            # the garbage collector wants to delete the executor, the
            # executor will finish first all calculations.
            # Hence, cancel all futures, so it will at least not start new
            # "tasks".
            #
            # For threads shutdown is intended to not terminate submitted
            # tasks.
            # For mp shutdown also doesn't work as desired and the processes
            # keep the program alive, i.e. it never stops properly.
            # Hence, use only cancel for both.
            # ex.shutdown(wait=False)
            #
            # ToDo: Changed in python version 3.9: Added cancel_futures.
            #  - Use ex.shutdown(`cancel_futures`), once we drop support for
            #    python 3.8.

            try:
                while True:
                    q.get(block=False).cancel()
            except queue.Empty:
                pass

    elif backend is False:

        @contextlib.contextmanager
        def PoolExecutor(max_workers):
            yield None

        def submit(ex, func, *args, **kwargs):
            return func(*args, **kwargs)

        def result(job):
            return job

        def terminate(ex, q):
            pass

    else:
        raise ValueError(backend)

    with PoolExecutor(max_workers) as executor:
        try:
            # First fill the buffer
            # If buffer full, take one element and push one new inside
            for ele in generator:
                if q.qsize() >= buffer_size:
                    yield result(q.get())
                q.put(submit(executor, function, ele, *args, **kwargs))
            while not q.empty():
                yield result(q.get())
        except GeneratorExit:
            # A GeneratorExit will not stop the PoolExecutor,
            # i.e. the PoolExecutor will finish all calculations,
            # before the PoolExecutor stops. This could take some time
            # and is useless.
            terminate(executor, q)
            raise


def single_thread_prefetch(
        generator: Iterable,
        buffer_size: int,
):
    """
    Iterate over the generator in a thread and yield the values.

    Why can this function provide a speedup, when it uses only one thread?
        The trick is, that it loads the next item, while the main thread can
        process the current example.

        Here an example, where `slow_function_1` and `slow_function_2` are
        executed in parallel:

            generator = (slow_function_1(i) for i in range(10))
            for item in single_thread_prefetch(generator, 2):
               slow_function_2(item) # Do something time consuming with item

        But when you do something like

            generator = (slow_function_1(i) for i in range(10))
            for ... in list(single_thread_prefetch(generator, 2)):
               slow_function_2(item) # Do something time consuming with item

        or

            generator = [slow_function_1(i) for i in range(10)]
            for ... in single_thread_prefetch(generator, 2):
               slow_function_2(item) # Do something time consuming with item

        you will observe no speedup.
        In the first example, the `list(...)` converts the output generator
        of this function to a `list` and in the second is a list comprehention
        used and not a generator comprehention (i.e. `(...)` -> `[...]`).

    Args:
        generator: Generator to iterate over
        buffer_size: Number of examples to buffer

    Returns:
        generator

    >>> generator = (print(i) for i in range(5))
    >>> list(single_thread_prefetch(generator, 2))
    0
    1
    2
    3
    4
    [None, None, None, None, None]
    >>> generator = (print(i) for i in range(10))
    >>> next(iter(single_thread_prefetch(generator, 2)))
    0
    1
    2
    >>> def foo(): raise RuntimeError("Thread Exception")
    >>> generator = (foo() for i in range(10))
    >>> for i in single_thread_prefetch(generator, 2):
    ...     print(i)
    Traceback (most recent call last):
    ...
    RuntimeError: Thread Exception

    """
    shutdown = False  # A "Lock" is not necessary
    data_queue = queue.Queue(buffer_size)
    unique_object = object()
    exc_info = None

    def worker():
        if shutdown:
            return
        try:
            for item in generator:
                if shutdown:
                    return
                data_queue.put(item)
                if shutdown:
                    return
        except Exception:
            # Save the exception and reraise it in the main thread
            nonlocal exc_info
            # https://stackoverflow.com/a/1854263/5766934
            exc_info = sys.exc_info()
        finally:
            data_queue.put(unique_object)

    thread = threading.Thread(target=worker, args=())
    thread.start()
    try:
        while True:
            item = data_queue.get()
            if item is unique_object:
                break
            else:
                yield item
    finally:
        shutdown = True
        try:
            # Handle a break of the iteration.
            # When the main thread decided to cancel the iteration, we must
            # trigger the thread to run in the shutdown, otherwise the thread
            # may hang in `data_queue.put(item)`
            # Not sure, if there is a critical timing, when multiple get calls
            # are necessary. Hence call it until it is empty.
            while True:
                data_queue.get_nowait()
        except queue.Empty:
            pass
        thread.join()
    if exc_info is not None:
        raise exc_info[1].with_traceback(exc_info[2])


if __name__ == '__main__':
    import time

    def identity(x):
        return x


    print(f'Expect: ' + ' '.join(map(str, range(10))))
    print('Got:    ', end='')
    for i in lazy_parallel_map(identity, range(10)):
        print(i, end=' ')
    print()

    print(f'Expect: ' + ' '.join(map(str, range(10))))
    print('Got:    ', end='')
    for backend in [
        't',
        'mp',
        'concurrent_mp',
        # False,
    ]:

        for i in lazy_parallel_map(identity, range(10), backend=backend):
            print(i, end=' ')
        print()


    def task(i):
        time.sleep(0.1)
        return i


    print(f'Expect: ' + ' '.join(map(str, range(10))))
    print('Got:    ', end='')
    for i in lazy_parallel_map(lambda x: x, range(10), backend='dill_mp'):
        print(i, end=' ')
    print()

    from paderbox.utils.timer import Timer
    t = Timer(verbose=True)

    print(f'Serial time: ')
    print('Got:    ', end='')
    with t:
        for i in lazy_parallel_map(task, range(10), backend='dill_mp',
                                   buffer_size=5, max_workers=2):
            print(i)
            # print(i, end=' ')

    # Does not work
    # for i in lazy_parallel_map(lambda x: x, range(10), backend='concurrent_mp'):
    #     print(i, end=' ')
    # print()
