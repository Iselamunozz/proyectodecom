import numpy as np

def rrc_filter(beta, sps, span):
    """
    Filtro Root Raised Cosine (RRC)
    beta : roll-off (0 a 1)
    sps  : samples per symbol (sobremuestreo)
    span : duración en símbolos
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):

        # ti = 0
        if np.isclose(ti, 0.0):
            h[i] = 1.0 + beta*(4/np.pi - 1)

        # ti = ± Ts/(4β)
        elif beta != 0 and np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )

        else:
            num = (np.sin(np.pi*ti*(1 - beta)) +
                   4*beta*ti*np.cos(np.pi*ti*(1 + beta)))
            den = (np.pi*ti*(1 - (4*beta*ti)**2))
            h[i] = num / (den + 1e-12)

    # Normalizar energía
    h = h / np.sqrt(np.sum(h**2))
    return h
