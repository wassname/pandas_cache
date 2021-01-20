# Pandas Cache  🐼 💸 


## Purpose
This module reduces loading times for resource-intensive pandas operations dramatically by memoizing the results of functions that return pandas DataFrames and Series. This can reduce over-dependence on jupyter notebooks for slow data operations.


The `@pd_cache` decorator function wraps a function that returns a pandas object. The included `@timeit` decorator is optional, and provides benchmarking times for a given function (does not execcute wrapped function more than once). The `del_cached()` function deletes all cached objects if desired.

## Installation
`pip3 install pandas_cache`

Note: currently only supports python3.6 or greater.

## Example Usage

```
from pandas_cache import pd_cache, timeit
import pandas as pd

@timeit
@pd_cache()
def time_consuming_dataframe_operation():
    # Make a large dictionary 
    x = {i: k for i, k in enumerate(range(2**16))}
    return pd.DataFrame([x])


time_consuming_dataframe_operation()
```

Output on first run:

```
	 > function time_consuming_dataframe_operation time: 2.5 s
	 | wrote .pd_cache/time_consuming_dataframe_operation_fe2bc4/ace6f4.phl
```

Output on second run:
```
	 | read .pd_cache/time_consuming_dataframe_operation_fe2bc4/ace6f4.hkl
	 > function time_consuming_dataframe_operation time: 6.0 ms
```
In this example, the 2.5 second operation has been memoized and the results are loaded in 6ms.

## How It Works
At runtime, the `@pd_cache` decorator :
* Takes the hash of the decorated function's plain text code, arguments, and keywords
* Pickles the pandas object returned by the decorated function
* Saves the pickle to a new `./.pd_cache/function_name_fe2bc4/a1b2c9.hkl` file, which includes hashes of function code and call arguments.

Upon running a second time the decorator:
* Hashes the function code again
* If the file already exists in the cache folder, it is loaded. 
* If the code in the function has changed in any way, the decorator use a new dir

## Caveats
* Only works with functions that return pandas objects with the `.to_pickle()` method.


