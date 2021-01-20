"""
Implements on disk caching of transformed dataframes

Used on a function that returns a single pandas object,
this decorator will execute the function, cache the dataframe as a pickle
file using the hash of function and subdirectory, and the arguments and filename.
The next time the function runs, if the hashes match what is on disk, the decoratored function will simply load and return
the pickled pandas object.

This can result in speedups of 10 to 100 times, or more, depending on the
complexity of the function that creates the dataframe.

The caveat is that previously cached dataframes will remain on disk.
"""

from functools import wraps

import pandas as pd
import pickle
import hashlib
import inspect
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def md5hash(s: str) -> str:
    return hashlib.md5(s).hexdigest()


def source_code(func) -> str:
    return "".join(inspect.getsourcelines(func)[0])


def pd_cache(cache_base: Path = Path(".pd_cache"), use_code: bool = True):
    def _pd_cache(func):
        @wraps(func)
        def cache(*args, **kw):
            # The subdirectory contains hahs of function name (and optionally code)
            if use_code:
                f_hash = md5hash(source_code(func).encode("utf-8"))[:6]
            else:
                f_hash = "-1"
            cache_dir = cache_base / f"{func.__name__}_{f_hash}"
            if not cache_dir.exists():
                cache_dir.mkdir(exist_ok=True, parents=True)
                logger.info(f"created `{cache_dir}` dir")

            # The file name contains the hash of functions args and kwargs
            key = pickle.dumps(args, 1) + pickle.dumps(kw, 1)
            hsh = md5hash(key)[:6]

            f = cache_dir / f"{hsh}.pkl.gz"

            if f.exists():
                df = pd.read_pickle(f)
                logger.debug(f"\t | read {f}")
                return df

            else:
                # Write new
                df = func(*args, **kw)
                df.to_pickle(f)
                logger.debug(f"\t | wrote {f}")
                return df

        return cache

    return _pd_cache
