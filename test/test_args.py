import numpy as np
import pandas as pd
from pandas_cache import pd_cache, timeit
import time
import logging
import shutil
from pathlib import Path
logging.basicConfig(level=logging.DEBUG)



class CustomObject:
    def __init__(self, a):
        self.a = a

def fun(n):
    return n

def make_args(n: int):
    """Make a range of datasets with a seed: n."""
    return [
        dict(b=[1, n], c=range(n)),
        set([n, 'a'*n, b'1']),
        (fun, n),
        range(n),
        np.arange(n),
        [0, n],
        (n, 10. * n, 'c'),
        pd.Series(n*2.0, index=list(range(n)), dtype="float32"),
        np.array([n] * 4, dtype="int32"),
        pd.Categorical(["test", "train", "test", "train"]*n),
        pd.Timestamp("20130102")-pd.Timedelta('1D') * n,
        CustomObject(n),
        # named_tuple(n,n*2,n*3)
    ]

def test_args(tmpdir):
    cache_dir = Path(tmpdir)

    def no_files():
        return len(list(cache_dir.glob('**/*.h5')))

    assert no_files()==0, 'cache_dir should be empty'

    @timeit
    @pd_cache(cache_base=cache_dir)
    def foo(*args, **kwargs):
        time.sleep(0.1)
        return pd.DataFrame(
            {
                "A": 1.0,
                "B": pd.Timestamp("20130102"),
                "C": pd.Series(1, index=list(range(4)), dtype="float32"),
                "D": np.array([3] * 4, dtype="int32"),
                "E": pd.Categorical(["test", "train", "test", "train"]),
                "F": "foo",
            }
        )

    print(tmpdir)

    print('should write')
    for i, arg in enumerate(make_args(1)):
        print(f'\tnfs={no_files()}, arg {i}: `{arg}` ')
        r = foo(arg, arg=arg)
    assert no_files()==12, 'cache_dir should be full'

    print('should read')
    for i, arg in enumerate(make_args(1)):
        print(f'\tnfs={no_files()}, arg {i}: `{arg}` ')
        r = foo(arg, arg=arg)
    assert no_files()==12, 'cache_dir should be same'

    print('should write again, since seed changed')
    for i, arg in enumerate(make_args(2)):
        print(f'\tnfs={no_files()}, arg {i}: `{arg}` ')
        r = foo(arg, arg=arg)
    assert no_files()==12*2, 'cache_dir should have 2x'
