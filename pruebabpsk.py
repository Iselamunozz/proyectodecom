import numpy as np
import matplotlib.pyplot as plt
from generadores.canalisi import canal_kumar_variable

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
snr_db = 1000   # Usa un valor razonable, 1000dB no tiene sentido físico

# ==============================
# 2. BITS Y MODULACIÓN BPSK
# ==============================
bits = np.random.randint(0, 2, Nbits)
symbols = 2*bits - 1

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
upsampled = np.zeros(len(symbols)*sps)
upsampled[::sps] = symbols
tx_bb = np.convolve(upsampled, rrc, mode='full')

# ==============================
# 5. MODULACIÓN PASABANDA
# ==============================
t = np.arange(len(tx_bb)) / fs
carrier = np.exp(1j * 2*np.pi*fc*t)
tx_pb = np.real(tx_bb * carrier)

# ==============================
# 6. CANAL KUMAR + RUIDO
# ==============================

rx_pb, h_chan = canal_kumar_variable(
    tx_signal=tx_pb,
    tipo_fading='Rayleigh',
    nivel_isi='nulo',
    snr_db=snr_db
)

# ==============================
# 7. DEMODULACIÓN COHERENTE
# ==============================
t = np.arange(len(rx_pb)) / fs
rx_mix = rx_pb * np.exp(-1j * 2*np.pi*fc*t)
rx_bb = np.convolve(rx_mix, rrc, mode='full')

# ==============================
# 8. ALINEACIÓN Y MUESTREO
# ==============================
delay_rrc = (len(rrc) - 1) // 2
delay_total = 2 * delay_rrc

rx_bb_aligned = rx_bb[delay_total:delay_total + len(upsampled)]
tx_bb_aligned = tx_bb[delay_rrc:delay_rrc + len(upsampled)]

# Normalización independiente
rx_bb_aligned /= np.max(np.abs(rx_bb_aligned))
tx_bb_aligned /= np.max(np.abs(tx_bb_aligned))

# Muestreo
muestras_simbolos = rx_bb_aligned[::sps]

# Corrección de fase (alineación al eje real)
fase_promedio = np.angle(np.mean(muestras_simbolos))
muestras_simbolos *= np.exp(-1j * fase_promedio)

# Normalización (opcional pero útil para constelación limpia)
muestras_simbolos /= np.max(np.abs(muestras_simbolos))

# Corrección de fase
muestras_simbolos *= np.exp(-1j * np.angle(np.mean(muestras_simbolos**2)))

# Decisión BPSK
decisions = (muestras_simbolos.real >= 0).astype(int)

# ==============================
# 9. BER
# ==============================
bits_rx = decisions[:len(bits)]
num_err = np.sum(bits_rx != bits)
ber = num_err / len(bits_rx)
print(f"\nBER = {ber:.3e}   Errores = {num_err}/{len(bits_rx)}")
print("Canal generado (Kumar):", h_chan)

# ==============================
# 10. GRÁFICAS
# ==============================

plt.figure(figsize=(10,5))
plt.plot(np.real(tx_bb_aligned[:800]), label='Transmitida')
plt.plot(np.real(rx_bb_aligned[:800]), '--', label='Recibida')
plt.title("Señal Banda Base (TX vs RX)")
plt.legend()
plt.grid()

plt.figure(figsize=(5,5))
plt.scatter(muestras_simbolos.real, muestras_simbolos.imag, alpha=0.5)
plt.title("Constelación BPSK")
plt.grid()
plt.axis('equal')

# Diagrama de ojo
num_trazas = 40
samples_per_eye = 2*sps
rx_real = np.real(rx_bb_aligned)
num_muestras = (len(rx_real)//samples_per_eye)*samples_per_eye
traces = rx_real[:num_muestras].reshape((-1, samples_per_eye))
plt.figure()
for i in range(min(num_trazas, len(traces))):
    plt.plot(traces[i], alpha=0.3)
plt.title("Diagrama de Ojo")
plt.grid()

# Respuesta del canal generado
plt.figure()
plt.stem(np.abs(h_chan))
plt.title("Magnitud de los taps (Canal Kumar)")
plt.grid()

plt.show()

