
#Fourier Analysis & FFT Spectral Verification
# Computes the frequency-domain representation of the BPSK modulated
# waveform using both the DTFT (manual summation) and FFT (numpy).
# Generates verification plots proving the signal energy is concentrated
# in the expected band around the carrier frequency Fc.

# DTFT Definition:
#   X(e^{j*omega}) = sum_{n=0}^{N-1} x[n] * e^{-j * omega * n}
#
# The DTFT is a continuous function of omega, while the N-point DFT
# (computed via FFT) samples it at omega_k = 2*pi*k/N.


import os
import numpy as np
import matplotlib.pyplot as plt


# CORE FUNCTIONS

def compute_dtft(x, omega_grid):
    """Evaluate the DTFT of a finite-length sequence on an arbitrary grid.

    The DTFT of x[n] (n = 0, 1, ..., N-1) at angular frequency omega is:

        X(e^{j*omega}) = sum_{n=0}^{N-1} x[n] * e^{-j * omega * n}

    This is computed via a matrix-vector product for efficiency while
    still being a direct implementation of the mathematical definition.

    Parameters
    ----------
    x : np.ndarray
        Input sequence of length N.
    omega_grid : np.ndarray
        Array of normalised angular frequencies (radians/sample).

    Returns
    -------
    X : np.ndarray, complex
        DTFT values at each point in omega_grid.
    """
    N = len(x)
    n = np.arange(N)

    # Build the analysis matrix: E[k, n] = e^{-j * omega_grid[k] * n}
    # Each row of E corresponds to one frequency bin.
    E = np.exp(-1j * np.outer(omega_grid, n))

    # Matrix-vector product gives the DTFT at all requested frequencies
    X = E @ x
    return X


def compute_fft_spectrum(x, fs):
    """Compute the single-sided magnitude spectrum using numpy's FFT.

    The FFT computes the N-point DFT, which equals the DTFT sampled at
    omega_k = 2*pi*k/N.  We take the positive-frequency half and
    normalise by the sequence length.

    Parameters
    ----------
    x : np.ndarray
        Time-domain signal of length N.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz (0 to Fs/2).
    magnitude : np.ndarray
        Normalised magnitude spectrum (single-sided).
    """
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)

    # Keep only non-negative frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = np.abs(X[pos_mask]) / N

    return freqs, magnitude


# PLOTTING

def plot_spectrum(freqs_fft, mag_fft, freqs_dtft, mag_dtft, fc, fs,
                  output_dir):
    """Plot FFT spectrum and overlay DTFT for comparison.

    Two subplots:
      Top:    Full spectrum (0 to Fs/2) from FFT
      Bottom: Zoomed view around the carrier with both DTFT and FFT
    """
    nyquist = fs / 2.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # op: Full FFT magnitude spectrum 
    axes[0].plot(freqs_fft / 1000, 20 * np.log10(mag_fft + 1e-12),
                 linewidth=0.5, color="C0")
    axes[0].axvline(fc / 1000, color="C3", linestyle="--", linewidth=1,
                    label=f"Fc = {fc:.0f} Hz")
    axes[0].axvline(nyquist / 1000, color="gray", linestyle=":",
                    linewidth=1, label=f"Fs/2 = {nyquist:.0f} Hz")
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Module 3-4 -- FFT Magnitude Spectrum (full band)")
    axes[0].set_xlim(0, nyquist / 1000)
    axes[0].set_ylim(-80, 5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #  Bottom: Zoomed around carrier, DTFT vs FFT 
    zoom_low = max(0, fc - 2000)
    zoom_high = fc + 2000
    mask_fft = (freqs_fft >= zoom_low) & (freqs_fft <= zoom_high)
    mask_dtft = (freqs_dtft >= zoom_low) & (freqs_dtft <= zoom_high)

    axes[1].plot(freqs_fft[mask_fft] / 1000,
                 20 * np.log10(mag_fft[mask_fft] + 1e-12),
                 linewidth=0.5, alpha=0.7, color="C0", label="FFT")
    axes[1].plot(freqs_dtft[mask_dtft] / 1000,
                 20 * np.log10(mag_dtft[mask_dtft] + 1e-12),
                 linewidth=1.0, color="C1", linestyle="--", label="DTFT")
    axes[1].axvline(fc / 1000, color="C3", linestyle="--", linewidth=1,
                    label=f"Fc = {fc:.0f} Hz")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("Module 3-4 -- Zoomed Spectrum: DTFT vs FFT")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fft_magnitude_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 3-4] Saved spectrum plot    -> {path}")
    return path


#  MODULE RUNNER 

def run(signal, fs, fc, output_dir):
    """Execute Fourier analysis: compute DTFT and FFT, generate plots.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain BPSK waveform from Module 2.
    fs : int
        Sampling rate in Hz.
    fc : float
        Carrier frequency in Hz.
    output_dir : str
        Directory for saving figures.

    Returns
    -------
    results : dict
        'freqs_fft', 'mag_fft', 'freqs_dtft', 'mag_dtft'
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 3-4 -- Fourier Analysis & FFT Spectral Verification")
    print("=" * 70)

    #Compute DTFT on a custom frequency grid 
    # We evaluate on 1024 points from 0 to pi (0 to Fs/2 in Hz).
    num_dtft_points = 1024
    omega_grid = np.linspace(0, np.pi, num_dtft_points, endpoint=False)
    freqs_dtft = omega_grid * fs / (2.0 * np.pi)   # convert to Hz

    print(f"  Computing DTFT on {num_dtft_points} frequency points...")
    X_dtft = compute_dtft(signal, omega_grid)
    mag_dtft = np.abs(X_dtft) / len(signal)

    # Compute FFT spectrum 
    print(f"  Computing FFT ({len(signal)}-point)...")
    freqs_fft, mag_fft = compute_fft_spectrum(signal, fs)

    #  Find the peak frequency 
    peak_idx = np.argmax(mag_fft)
    peak_freq = freqs_fft[peak_idx]
    peak_mag_db = 20 * np.log10(mag_fft[peak_idx] + 1e-12)

    print(f"\n  Spectral analysis results:")
    print(f"    Peak frequency:  {peak_freq:.1f} Hz  (expected ~{fc:.0f} Hz)")
    print(f"    Peak magnitude:  {peak_mag_db:.1f} dB")
    print(f"    Signal is band-limited to [{fc - 500:.0f}, {fc + 500:.0f}] Hz approx.")

    # -Generate plots 
    plot_spectrum(freqs_fft, mag_fft, freqs_dtft, mag_dtft, fc, fs,
                  output_dir)

    return {
        "freqs_fft":  freqs_fft,
        "mag_fft":    mag_fft,
        "freqs_dtft": freqs_dtft,
        "mag_dtft":   mag_dtft,
    }
