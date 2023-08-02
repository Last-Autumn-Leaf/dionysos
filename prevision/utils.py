import time
from datetime import timedelta
import functools
from pathlib import Path


def timeThis(message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(message, timedelta(seconds=end - start))
            return result

        return wrapper

    return decorator


@timeThis("Execution time: ")
def test(a='test'):
    print(a)


from contextlib import contextmanager


@contextmanager
def timeThat(name='', storage=None):
    try:
        start = time.time()
        yield ...
    finally:
        end = time.time()
        print(name + ' finished in ', timedelta(seconds=end - start))
        if storage != None:
            storage['completed_time'] = int((end - start) / 60)


project_name = "dionysos"


def setProjectpath():
    project_dir = Path.cwd()
    while project_dir.name != project_name:
        project_dir = project_dir.parent
        if project_dir.parent == project_dir:
            ValueError(f"Le dossier parent '{project_name}' n'a pas été trouvé.")
            print("project directory not found")
            break
    return project_dir
