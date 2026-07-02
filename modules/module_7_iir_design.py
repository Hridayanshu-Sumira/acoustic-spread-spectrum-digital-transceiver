# ---------------------------------------------------------------------------
# module_7_iir_design.py -- IIR Noise-Cancellation Filter Design
#
# Designs a digital Butterworth bandpass filter to isolate the BPSK
# carrier band from broadband channel noise. The actual filtering is 
# performed using the Direct Form II Transposed implementation in Module 6.
#
# Key DSP concepts:
#   - Analog filter prototypes (Butterworth)
#   - Bilinear Transform (BLT) for analog-to-digital mapping
#   - IIR filter difference equations
# ---------------------------------------------------------------------------

import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# Import our custom DF2T filter engine
from modules.module_6_filter_structures import direct_form_2_transposed


def design_butterworth_bandpass(order, f_low, f_high, fs):
    """Design a Butterworth bandpass filter using the Bilinear Transform.

    Uses scipy.signal.butter to compute the transfer function
    coefficients b and a. This uses the BLT under the hood.

    Parameters
    ----------
    order : int
        Filter order per side.
    f_low : float
        Lower passband edge in Hz.
    f_high : float
        Upper passband edge in Hz.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.
    """
    nyquist = fs / 2.0
    low = f_low / nyquist
    high = f_high / nyquist

    # Generate digital filter coefficients (b, a)
    # The 'bandpass' btype defaults to analog=False, which means it uses
    # the Bilinear Transform for continuous-to-discrete mapping.
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
    return b, a


def apply_iir_filter(b, a, signal):
    """Apply an IIR filter using Module 6's Direct Form II Transposed.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.
    signal : np.ndarray
        Input signal to filter.

    Returns
    -------
    filtered : np.ndarray
        Filtered output signal.
    """
    # Use our custom DF2T engine (no scipy.signal.lfilter)
    filtered = direct_form_2_transposed(b, a, signal)
    return filtered


def plot_iir_response(b, a, fs, output_dir):
    """Plot the frequency response of the designed IIR filter."""
    # Compute frequency response (evaluating DTFT of h[n])
    w, h = scipy.signal.freqz(b, a, worN=1024)
    freqs = w * fs / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Magnitude response
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    axes[0].plot(freqs, mag_db, 'C0')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Module 7 -- IIR Bandpass Filter Frequency Response')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-80, 5)

    # Phase response
    phase = np.unwrap(np.angle(h))
    axes[1].plot(freqs, phase, 'C1')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "iir_frequency_response.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 7] Saved IIR response plot  -> {path}")
    return path


def run(signal, order, f_low, f_high, fs, output_dir):
    """Execute Module 7: design bandpass IIR and filter the corrupted signal."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 7 -- IIR Noise-Cancellation Filter Design")
    print("=" * 70)

    # ---- Step 1: Design filter ----
    print(f"  Designing Butterworth Bandpass Filter (order={order})")
    print(f"  Passband: {f_low} Hz - {f_high} Hz")
    b, a = design_butterworth_bandpass(order, f_low, f_high, fs)

    print(f"  b coefficients: {b.round(4)}")
    print(f"  a coefficients: {a.round(4)}")

    # ---- Step 2: Apply filter manually using Module 6 DF2T ----
    print(f"  Applying filter using custom DF2T engine...")
    filtered_signal = apply_iir_filter(b, a, signal)

    # Compute SNR improvement
    # Note: rigorous SNR calc needs knowing the exact clean signal alignment,
    # but we can measure power before and after.
    # For now, just logging max amplitude to ensure stability.
    print(f"  Filtered signal max amplitude: {np.max(np.abs(filtered_signal)):.4f}")

    # ---- Step 3: Plots ----
    plot_iir_response(b, a, fs, output_dir)

    return {
        "filtered_signal": filtered_signal,
        "b": b,
        "a": a
    }
