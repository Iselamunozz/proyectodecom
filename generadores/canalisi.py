import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def canal_kumar_variable(tx_signal, tipo_fading='Rayleigh', nivel_isi='bajo', snr_db=20):
    """
    Simula el canal descrito por Kumar et al. (Eq. 4 y Tabla 1).
    Permite variar la intensidad del ISI modificando el perfil de retardo.
    
    Args:
        tx_signal: Señal de entrada (muestras complejas).
        tipo_fading: 'Rayleigh' (sin línea de vista) o 'Rician' (con línea de vista, K=10).
        nivel_isi: 'nulo', 'bajo', 'medio', 'alto'. Controla la dispersión de retardo.
        snr_db: Relación Señal a Ruido en dB.
        
    Returns:
        rx_signal: Señal recibida con ISI y Ruido.
        h: La respuesta al impulso del canal generado (para análisis).
    """
    
    # 1. DEFINICIÓN DEL PERFIL DE RETARDO (CONTROL DE ISI)
    # Basado en la lógica de "Power Delay Profile" de la ITU-R mencionada en el paper.
    # Más taps y decaimiento más lento = Más ISI.
    
    if nivel_isi == 'nulo':
        n_taps = 1
        decay_factor = 0 # No importa, solo hay 1 tap
    elif nivel_isi == 'bajo':
        n_taps = 3
        decay_factor = 2.0 # Decae muy rápido (ecos débiles)
    elif nivel_isi == 'medio':
        n_taps = 6
        decay_factor = 1.0 # Decae normal
    elif nivel_isi == 'alto':
        n_taps = 12
        decay_factor = 0.3 # Decae muy lento (ecos fuertes y lejanos = ISI severo)
    else:
        raise ValueError("Nivel de ISI no reconocido")

    # 2. GENERACIÓN DE COEFICIENTES (TAPS)
    # Generamos la envolvente de potencia (Power Delay Profile)
    pdp = np.exp(-np.arange(n_taps) * decay_factor)
    
    # Generamos la parte aleatoria (dispersión Rayleigh para todos los taps)
    # Componente Gaussiana compleja con media 0 y varianza 1/2 por eje
    rayleigh_component = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) * np.sqrt(0.5)
    
    # Aplicamos el perfil de potencia a los componentes aleatorios
    h = rayleigh_component * np.sqrt(pdp)
    
    # 3. LÓGICA RICIAN (Factor K=10 según Tabla 1 de Kumar)
    # Si es Rician, el primer tap (h[0]) tiene una componente fuerte de Línea de Vista (LOS)
    if tipo_fading == 'Rician':
        K = 10 # Valor extraído explícitamente del paper 
        # Potencia LOS vs NLOS
        p_los = np.sqrt(K / (K + 1))
        p_nlos = np.sqrt(1 / (K + 1))
        
        # El primer tap se modifica: Componente Fija + Componente Aleatoria
        h[0] = p_los + p_nlos * h[0]
        # Los demás taps siguen siendo Rayleigh (rebotes) escalados
    
    # Normalizar la energía del canal a 1 para mantener la consistencia del SNR
    h = h / np.linalg.norm(h)

    # 4. APLICACIÓN DEL ISI (Ec. 4 de Kumar - Convolución) [cite: 2580]
    rx_isi = signal.convolve(tx_signal, h, mode='same')
    
    # 5. AGREGAR RUIDO (AWGN)
    sig_power = np.mean(np.abs(rx_isi)**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise = (np.random.randn(len(rx_isi)) + 1j*np.random.randn(len(rx_isi))) * np.sqrt(noise_power/2)
    
    rx_final = rx_isi + noise
    
    return rx_final, h