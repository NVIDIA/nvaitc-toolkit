from model.loader import loaders
from model.util import find_class_in_mod

# from loaders import *

def get_loader(name, batch_size, rank, ngpu, path, **kwargs):
    loadercls = find_class_in_mod(loaders, name)
    print(kwargs)
    return loadercls(batch_size, rank, ngpu, path, **kwargs)

