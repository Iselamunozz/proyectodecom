import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fftshift, fft
from generadores.canalisi import generate_canalisi

# Parámetros de la simulación
N = 1000  # Número de bits
sps = 8   # Muestras por símbolo
EbN0_dB = 1000  # Eb/N0 en dB para el canal con ruido
ISI_taps = 4  # Coeficientes del canal ISI

# 1. Generar bits aleatorios y BPSK (0 -> -1, 1 -> 1)
bits = np.random.randint(0, 2, N)
symbols = 2*bits - 1  # BPSK mapping

# 2. Pulso rectangular (dirac a sps)
symbols_upsampled = np.zeros(len(symbols) * sps)
symbols_upsampled[::sps] = symbols

# 3. Filtro de modelado: Raised Cosine (pasa banda)
rolloff = 0.35
num_taps = 101
t, h_rc = signal.kaiserord(60, rolloff / sps)
h_rc = signal.firwin(num_taps, 1/sps, window='hamming')

# Señal transmitida (ideal)
tx_signal = np.convolve(symbols_upsampled, h_rc, mode='same')

# 4. Canal con ISI
channel = generate_canalisi(length=4, attenuation=0-3)
channel[::sps] = ISI_taps  # ISI entre símbolos
rx_signal_isi = np.convolve(tx_signal, channel, mode='same')

# 5. Agregar ruido AWGN
EbN0 = 10**(EbN0_dB/10)
Es = np.sum(np.abs(tx_signal)**2) / len(tx_signal)
N0 = Es / EbN0
noise = np.sqrt(N0/2) * np.random.randn(len(rx_signal_isi))
rx_signal_isi_awgn = rx_signal_isi + noise

# Función para graficar espectro
def plot_spectrum(signal, sps, title):
    f = np.linspace(-0.5, 0.5, len(signal))
    spectrum = fftshift(np.abs(fft(signal)))
    plt.plot(f, 20*np.log10(spectrum/np.max(spectrum)))
    plt.title(title)
    plt.xlabel("Frecuencia normalizada")
    plt.ylabel("Magnitud (dB)")
    plt.grid()

# Función para graficar constelación
def plot_constellation(signal, sps, title):
    # Tomar muestras en cada símbolo (asumiendo sincronía perfecta)
    samples = signal[::sps]
    plt.plot(samples.real, samples.imag, 'o')
    plt.title(title)
    plt.grid()
    plt.xlabel("I")
    plt.ylabel("Q")

# 6. Visualizaciones
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(tx_signal[:500])
plt.title("Señal en el tiempo (Ideal)")
plt.grid()

plt.subplot(3, 2, 2)
plot_spectrum(tx_signal, sps, "Espectro señal (Ideal)")

plt.subplot(3, 2, 3)
plt.plot(rx_signal_isi_awgn[:500])
plt.title("Señal en el tiempo (Canal ISI + AWGN)")
plt.grid()

plt.subplot(3, 2, 4)
plot_spectrum(rx_signal_isi_awgn, sps, "Espectro señal (ISI + AWGN)")

plt.subplot(3, 2, 5)
plot_constellation(tx_signal, sps, "Constelación (Ideal)")

plt.subplot(3, 2, 6)
plot_constellation(rx_signal_isi_awgn, sps, "Constelación (ISI + AWGN)")

plt.tight_layout()
plt.show()
