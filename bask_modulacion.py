import numpy as np
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS
# ============================
N_bits      = 200           # número de bits
sps         = 100           # samples per symbol (sobremuestreo)
Rb          = 1e3           # tasa de símbolos (1 ksym/s)
fs          = Rb * sps      # frecuencia de muestreo
fc          = 20e3          # frecuencia de portadora (20 kHz)
snr_db      = 100            # SNR en dB
beta        = 0.25          # roll-off del filtro RRC
span        = 8             # duración del filtro en símbolos
np.random.seed(0)           # para reproducibilidad

# ============================
# FUNCIONES AUXILIARES
# ============================
def rrc_filter(beta, sps, span):
    """
    Filtro Root Raised Cosine (RRC)
    beta: roll-off
    sps: samples per symbol
    span: duración en símbolos
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps  # en tiempos de símbolo

    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            # caso t = 0
            h[i] = 1.0 - beta + 4*beta/np.pi
        elif abs(abs(4*beta*ti) - 1.0) < 1e-8:
            # caso singular |4*beta*t| = 1
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = (np.sin(np.pi*ti*(1-beta)) +
                   4*beta*ti*np.cos(np.pi*ti*(1
