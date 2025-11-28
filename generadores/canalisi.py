import numpy as np
from scipy import signal

def canal_kumar_variable(tx_signal, tipo_fading='Rayleigh', nivel_isi='bajo', snr_db=20):
    """
    Simula un canal con ISI y fading (Rayleigh, Rician o nulo),
    con control sobre el nivel de ISI y fase limitada.

    Parámetros:
        tx_signal: Señal de entrada (compleja o real)
        tipo_fading: 'nulo', 'Rayleigh' o 'Rician'
        nivel_isi: 'nulo', 'bajo', 'medio', 'alto'
        snr_db: SNR en decibelios

    Retorna:
        rx_signal: Señal con canal y ruido
        h: respuesta al impulso del canal
    """

    def limitar_fase(complejo, max_fase=np.pi/4):
        """Limita la fase de un número complejo a ±max_fase radianes"""
        mag = np.abs(complejo)
        ang = np.angle(complejo)
        ang_limitado = np.clip(ang, -max_fase, max_fase)
        return mag * np.exp(1j * ang_limitado)

    # ===== PERFIL DE RETARDO =====
    if nivel_isi == 'nulo':
        h = np.array([1.0 + 0j])  # Canal ideal
    else:
        if nivel_isi == 'bajo':
            n_taps = 3
            decay_factor = 2.0
        elif nivel_isi == 'medio':
            n_taps = 6
            decay_factor = 1.0
        elif nivel_isi == 'alto':
            n_taps = 12
            decay_factor = 0.3
        else:
            raise ValueError("Nivel ISI no reconocido")

        # Perfil de potencia (PDP)
        pdp = np.exp(-np.arange(n_taps) * decay_factor)

        # Taps iniciales (Rayleigh)
        taps = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) * np.sqrt(0.5)
        h = taps * np.sqrt(pdp)

        # Limitar la fase a ±45°
        h = np.array([limitar_fase(c, max_fase=np.pi/4) for c in h])

        # Si es Rician, se modifica el primer tap
        if tipo_fading == 'Rician':
            K = 10
            p_los = np.sqrt(K / (K + 1))
            p_nlos = np.sqrt(1 / (K + 1))
            h[0] = p_los + p_nlos * h[0]
            h[0] = limitar_fase(h[0], max_fase=np.pi/4)

    # Normalizar canal (potencia unitaria)
    h = h / np.linalg.norm(h)

    # ===== CONVOLUCIÓN (ISI) =====
    rx_isi = signal.convolve(tx_signal, h, mode='same')

    # ===== RUIDO =====
    signal_power = np.mean(np.abs(rx_isi)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = (np.random.randn(len(rx_isi)) + 1j*np.random.randn(len(rx_isi))) * np.sqrt(noise_power / 2)

    rx_final = rx_isi + noise

    return rx_final, h
