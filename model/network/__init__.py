from model.network import networks
from model.util import find_class_in_mod

def get_network(arch):
    nncls = find_class_in_mod(networks, arch)
    return nncls()

