import time

def find_class_in_mod(mod, name):
    for c in dir(mod):
        a = getattr(mod, c)
        if hasattr(a, 'name') and getattr(a, 'name') == name:
            return a

    raise KeyError('Cannot find class')

def timeme(method):
    def timer(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("{0}, {1} ms".format(method.__name__, (te - ts)*1000))
        return result
    return timer

