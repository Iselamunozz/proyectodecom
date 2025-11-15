import numpy as np

def cos_elevado(sps, rolloff, span):
    """
    Filtro Raised Cosine (coseno elevado)
    sps: samples per symbol
    rolloff: factor de rolloff (0 <= α <= 1)
    span: duración en símbolos (ej. 6)
    """
    N = sps * span + 1
    t = np.arange(-span/2, span/2 + 1/sps, 1/sps)
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0
        elif abs(abs(4 * rolloff * ti) - 1) < 1e-8:
            # Caso singular: t = ±1/(4α)
            h[i] = (np.pi/4) * np.sinc(1/(2*rolloff))
        else:
            num = np.sin(np.pi * ti * (1 - rolloff)) + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
            den = np.pi * ti * (1 - (4 * rolloff * ti)**2)
            h[i] = num / den

    h /= np.sum(h)  # normalización unitaria
    return h
def eliminar_delay(signal, h):
    delay = (len(h) - 1) // 2
    return signal[delay:-delay]

