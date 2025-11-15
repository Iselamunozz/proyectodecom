import numpy as np

def generate_canalisi(length=3, attenuation=0.5):
    """
    Genera un canal FIR con ISI decreciente.
    length: número de taps
    attenuation: cuánto decaen los ecos (entre 0 y 1)
    """
    h = [attenuation**i for i in range(length)]
    return np.array(h) / np.sum(h)
