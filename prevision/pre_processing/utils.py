import time
from datetime import timedelta
import functools
def timeThis(message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(message, timedelta(seconds=end-start))
            return result
        return wrapper
    return decorator

@timeThis("Execution time: ")
def test(a='test'):
    print(a)



from contextlib import contextmanager
@contextmanager
def timeThat(name='',storage=None):
    try:
        start = time.time()
        yield ...
    finally:
        end = time.time()
        print(name+' finished in ',timedelta(seconds=end-start))
        if storage != None :
            storage['completed_time']= int((end-start)/60 )