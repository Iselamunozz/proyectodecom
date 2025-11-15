import matplotlib.pyplot as plt
import numpy as np
import os

def graficar_constelacion(muestras, nombre_archivo):
     os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
     plt.figure(figsize=(4, 4))
     
     plt.scatter(muestras.real, muestras.imag, s=10, alpha=0.2)
     #ejes
     plt.axvline(0, color = 'black', linewidth=0.7)
     plt.axhline(0, color = 'black', linewidth = 0.7)
     plt.axis('equal')
     plt.xticks([]) #eliminar los valores en el eje x
     plt.xticks([]) # eliminar los valores del eje y
     plt.grid(False)
     plt.tight_layout(pad=0)
     plt.savefig(nombre_archivo,bbox_inches = 'tight', pad_inches = 0)
     plt.savefig(nombre_archivo, dpi=300)
     plt.close()
    
def graficar_espectro(senal, nombre_archivo):
     os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
     plt.figure(figsize=(4, 3))
     
     # Cálculo del espectro
     freqs = np.fft.fftfreq(len(senal), d=1)
     spectrum = np.fft.fft(senal)
     plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(20 * np.log10(np.abs(spectrum) + 1e-12)))

    # Eje cero
     plt.axhline(0, color='black', linewidth=0.5)
     plt.axvline(0, color='black', linewidth=0.5)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.box(False)
     plt.tight_layout(pad=0)
     plt.savefig(nombre_archivo, bbox_inches='tight', pad_inches=0)
     plt.close()


    
def graficar_tiempo(senal, nombre_archivo, limite=500):
    os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
    plt.figure(figsize=(6, 2.5))
    plt.plot(senal[:limite], linewidth=0.8)

    # Eje horizontal (amplitud 0)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.box(False)
    plt.tight_layout(pad=0)
    plt.savefig(nombre_archivo, bbox_inches='tight', pad_inches=0)
    plt.close()

