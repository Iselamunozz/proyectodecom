import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, fir_filter_design, freqz
from scipy.signal.windows import hann
from scipy.fftpack import fft, fftshift

# Parametros del diseño
simbolos_manuales = [0 1 0 1 0 1 0 1 0 1 1 1 0 0 0]
sobremuestro = 10
snr_db = 100
atenuacion_isi = 0.5

