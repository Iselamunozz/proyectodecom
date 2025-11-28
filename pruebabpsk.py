import numpy as np
import matplotlib.pyplot as plt
from generadores.canalisi import canal_kumar_variable
from scipy.signal import convolve

# ==============================
# 1. PARÁMETROS DEL SISTEMA
# ==============================
Nbits = 200
sps = 30
Rb = 1000
fs = Rb * sps
fc = 5000
beta = 0.25
span = 30
snr_db = 1000

# ==============================
# 2. BITS Y MODULACIÓN BPSK
# ==============================
bits = np.random.randint(0, 2, Nbits)
symbols = 2 * bits - 1

# ==============================
# 3. FILTRO RRC
# ==============================
def rrc_filter(beta, sps, span):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + (4*beta/np.pi)
        elif np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = (np.sin(np.pi*ti*(1-beta)) +
                   4*beta*ti*np.cos(np.pi*ti*(1+beta)))
            den = (np.pi*ti*(1-(4*beta*ti)**2))
            h[i] = num / den

    return h / np.sqrt(np.sum(h**2))

rrc = rrc_filter(beta, sps, span)

# ==============================
# 4. SOBREMUESTREO + FILTRADO TX
# ==============================
upsampled = np.zeros(len(symbols) * sps)
upsampled[::sps] = symbols
tx_bb = np.convolve(upsampled, rrc, mode='full')

# ==============================
# 5. MODULACIÓN PASABANDA
# ==============================
t = np.arange(len(tx_bb)) / fs
carrier = np.exp(1j * 2 * np.pi * fc * t)
tx_pb = np.real(tx_bb * carrier)

# ==============================
# 6. CANAL (Kumar + ISI + Ruido)
# ==============================
rx_pb, h_chan = canal_kumar_variable(
    tx_signal=tx_pb,
    tipo_fading='Rician',
    nivel_isi='nulo',
    snr_db=snr_db,
    max_fase=np.pi/16
)

# ==============================
# 7. DEMODULACIÓN COHERENTE
# ==============================
rx_mix = rx_pb * np.exp(-1j * 2 * np.pi * fc * t)
rx_bb = np.convolve(rx_mix, rrc, mode='full')

# ==============================
# 8. ALINEACIÓN Y CORRECCIÓN DE FASE
# ==============================
# ==============================
# 8. ALINEACIÓN Y CORRECCIÓN DE FASE (CORREGIDO)
# ==============================
# 1. Calcular el retardo de los filtros (RRC TX + RRC RX)
delay_rrc = (len(rrc) - 1) // 2
delay_total = 2 * delay_rrc

# 2. Recortar para quitar los transitorios de los filtros
# Nota: Usamos slice seguro para evitar errores de índice
start_idx = delay_total
end_idx = delay_total + len(upsampled)
rx_bb_aligned = rx_bb[start_idx : end_idx]
tx_bb_aligned = tx_bb[delay_rrc : delay_rrc + len(upsampled)]

# 3. CORRECCIÓN DE FASE FINA (Algoritmo de Potencia par BPSK)
# Elevamos al cuadrado para quitar la modulación (0->0, 180->360)
signal_squared = rx_bb_aligned**2
doble_error_fase = np.angle(np.mean(signal_squared)) # Esto es 2*theta

# IMPORTANTE: Dividir por 2 para obtener el error real
error_fase = doble_error_fase / 2.0

# Aplicar la corrección
rx_bb_aligned = rx_bb_aligned * np.exp(-1j * error_fase)

# 4. CORRECCIÓN DE AMBIGÜEDAD DE 180 GRADOS
# Como elevamos al cuadrado, no sabemos si corregimos hacia 0 o hacia 180.
# Usamos la señal conocida (tx_bb) para verificar la orientación.
proyeccion = np.real(np.vdot(rx_bb_aligned, tx_bb_aligned))

if proyeccion < 0:
    print("⚠️ Ambigüedad de fase detectada: Rotando 180°")
    rx_bb_aligned *= -1

# === Normalización final ===
rx_bb_aligned /= np.max(np.abs(rx_bb_aligned))
tx_bb_aligned /= np.max(np.abs(tx_bb_aligned))

# ==============================
# 9. MUESTREO Y DECISIÓN
# ==============================
muestras_simbolos = rx_bb_aligned[::sps]
decisions = (muestras_simbolos.real >= 0).astype(int)

# ==============================
# 10. BER
# ==============================
bits_rx = decisions[:len(bits)]
num_err = np.sum(bits_rx != bits)
ber = num_err / len(bits_rx)
print(f"\nBER = {ber:.3e}   Errores = {num_err}/{len(bits_rx)}\n")

# ==============================
# 11. GRÁFICAS
# ==============================

# Comparación TX vs RX
plt.figure(figsize=(10,5))
plt.plot(np.real(tx_bb_aligned[:800]), label='Transmitida')
plt.plot(np.real(rx_bb_aligned[:800]), '--', label='Recibida')
plt.title("Señal Banda Base (TX vs RX)")
plt.xlabel("Muestra")
plt.ylim(-1.1, 1.1)
plt.legend()
plt.grid()

# Constelación
plt.figure(figsize=(5,5))
plt.scatter(muestras_simbolos.real, muestras_simbolos.imag, alpha=0.5)
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.title("Constelación BPSK")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.grid()
plt.axis('equal')

# Diagrama de Ojo
num_trazas = 50
samples_per_eye = 2 * sps
rx_real = np.real(rx_bb_aligned)
num_muestras = (len(rx_real) // samples_per_eye) * samples_per_eye
traces = rx_real[:num_muestras].reshape((-1, samples_per_eye))

plt.figure(figsize=(6,4))
for i in range(min(num_trazas, len(traces))):
    plt.plot(traces[i], alpha=0.3)
plt.title("Diagrama de Ojo")
plt.grid()

# Espectro
from numpy.fft import fft, fftfreq, fftshift
Nfft = 4096
TXF = fftshift(fft(tx_pb[:Nfft]))
freqs = fftshift(fftfreq(Nfft, d=1/fs))

plt.figure(figsize=(7,4))
plt.plot(freqs/1000, 20*np.log10(np.abs(TXF) + 1e-12))
plt.title("Espectro de la señal PASABANDA")
plt.xlabel("Frecuencia [kHz]")
plt.ylabel("Magnitud [dB]")
plt.grid()

# Respuesta total del sistema
h_total = convolve(convolve(rrc, h_chan), rrc)
plt.figure()
plt.plot(h_total)
plt.title("Respuesta al impulso total del sistema")
plt.grid()

# Pasabanda TX vs RX
plt.figure(figsize=(10, 4))
plt.plot(tx_pb[:1000], label='TX Pasabanda')
plt.plot(rx_pb[:1000], '--', label='RX Pasabanda', alpha=0.7)
plt.title("Señal Pasabanda Transmitida vs Recibida")
plt.xlabel("Muestra")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

plt.show()
