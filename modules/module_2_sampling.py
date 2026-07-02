
# Sampling and Continuous Waveform Synthesis
# Takes the binary bitstream x[n] from Module 1 and modulates it onto
# a sinusoidal carrier using Binary Phase Shift Keying (BPSK).
#
# Key DSP concepts demonstrated:
#   - The Nyquist-Shannon Sampling Theorem
#   - Discrete-time representation of a band-limited analog waveform
#   - BPSK modulation: bit 1 -> +cos(2*pi*Fc*t)
#                      bit 0 -> -cos(2*pi*Fc*t)
#   - Relationship between sampling rate Fs, carrier Fc, and alias-free
#     reconstruction

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#  CORE FUNCTIONS

def verify_nyquist(fc, fs):
    """Check that the carrier frequency satisfies the Nyquist criterion.

    Nyquist-Shannon Sampling Theorem:
        To avoid aliasing, the sampling rate Fs must be strictly greater
        than twice the highest frequency component in the signal.

            Fs > 2 * Fmax

        For a pure BPSK carrier at frequency Fc, the highest spectral
        component is Fc itself (ignoring the negligible bandwidth of the
        rectangular bit envelope), so we need:

            Fs > 2 * Fc
        or equivalently:
            Fc < Fs / 2   (the folding frequency)

    Parameters
    ----------
    fc : float
        Carrier frequency in Hz.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    compliant : bool
        True if the Nyquist condition is satisfied.
    """
    nyquist_freq = fs / 2.0
    compliant = fc < nyquist_freq

    print(f"\n  Nyquist Sampling Theorem check:")
    print(f"    Carrier  Fc     = {fc:.1f} Hz")
    print(f"    Sampling Fs     = {fs} Hz")
    print(f"    Folding  Fs/2   = {nyquist_freq:.1f} Hz")
    print(f"    Fc < Fs/2 ?     {'YES -- No aliasing' if compliant else 'NO -- ALIASING RISK!'}")
    print(f"    Headroom:       {nyquist_freq - fc:.1f} Hz")

    if not compliant:
        raise ValueError(
            f"Nyquist violation: Fc={fc} Hz >= Fs/2={nyquist_freq} Hz. "
            "Increase Fs or decrease Fc."
        )
    return compliant


def bpsk_modulate(bitstream, fs, fc, samples_per_bit):
    """Modulate a binary bitstream onto a cosine carrier using BPSK.

    Binary Phase Shift Keying (BPSK) maps each bit to a fixed-duration
    burst of a sinusoidal carrier:

        bit = 1  -->  s(t) = +A * cos(2 * pi * Fc * t)    (phase = 0)
        bit = 0  -->  s(t) = -A * cos(2 * pi * Fc * t)    (phase = pi)

    The amplitude A is set to 1.0 (unit amplitude) for simplicity.

    In discrete time, t is replaced by  t = n / Fs  where n is the
    sample index.  Each bit occupies exactly `samples_per_bit` samples,
    giving a bit rate of  Rb = Fs / samples_per_bit  bits per second.

    Parameters
    ----------
    bitstream : np.ndarray, dtype int
        The binary sequence (0s and 1s) from Module 1.
    fs : int
        Sampling rate in Hz.
    fc : float
        Carrier frequency in Hz.
    samples_per_bit : int
        Number of discrete samples that represent one bit duration.

    Returns
    -------
    t : np.ndarray, dtype float
        Time vector in seconds, length = len(bitstream) * samples_per_bit.
    signal : np.ndarray, dtype float
        The BPSK-modulated waveform, same length as t.
    """
    num_bits = len(bitstream)
    total_samples = num_bits * samples_per_bit

    # Build the complete time vector for the whole message
    t = np.arange(total_samples) / fs

    # Pre-allocate the output waveform
    signal = np.zeros(total_samples, dtype=float)

    # Modulate each bit one by one
    for i, bit in enumerate(bitstream):
        start = i * samples_per_bit
        end   = start + samples_per_bit

        # Time indices for this particular bit slot
        t_bit = t[start:end]

        # BPSK mapping: bit 1 -> +cos, bit 0 -> -cos
        # Equivalently, we can write  amplitude = 2*bit - 1  which maps
        # {0, 1} to {-1, +1}.
        amplitude = 2 * bit - 1
        signal[start:end] = amplitude * np.cos(2.0 * np.pi * fc * t_bit)

    return t, signal


#PLOTTING

def plot_modulated_waveform(t, signal, fs, fc, samples_per_bit, bitstream,
                            output_dir):
    """Save the full BPSK waveform and a zoomed-in view of the first bits."""

    #Full waveform 
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t * 1000, signal, linewidth=0.4, color="C0")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Module 2 -- Full BPSK Modulated Waveform")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path_full = os.path.join(output_dir, "bpsk_waveform_full.png")
    fig.savefig(path_full, dpi=150)
    plt.close(fig)
    print(f"  [Module 2] Saved full waveform      -> {path_full}")

    # Zoomed view: first 4 bits
    num_zoom_bits = min(4, len(bitstream))
    zoom_samples = num_zoom_bits * samples_per_bit
    t_zoom = t[:zoom_samples] * 1000        # convert to ms
    s_zoom = signal[:zoom_samples]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_zoom, s_zoom, linewidth=0.8, color="C0")

    # Mark bit boundaries with vertical dashed lines
    for b in range(num_zoom_bits + 1):
        boundary_ms = b * samples_per_bit / fs * 1000
        ax.axvline(boundary_ms, color="gray", linestyle="--", linewidth=0.6,
                   alpha=0.7)
    # Label each bit above the waveform
    for b in range(num_zoom_bits):
        center_ms = (b + 0.5) * samples_per_bit / fs * 1000
        ax.text(center_ms, 1.15, str(bitstream[b]),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="C3")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Module 2 -- BPSK Zoomed (first {num_zoom_bits} bits)")
    ax.set_ylim(-1.4, 1.4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path_zoom = os.path.join(output_dir, "bpsk_waveform_zoomed.png")
    fig.savefig(path_zoom, dpi=150)
    plt.close(fig)
    print(f"  [Module 2] Saved zoomed waveform    -> {path_zoom}")

    return path_full, path_zoom


def plot_nyquist_diagram(fc, fs, output_dir):
    """Draw a simple frequency axis showing Fc relative to the Nyquist limit.

    This is a conceptual diagram, not a computed spectrum.  It visually
    proves that the carrier sits safely below the folding frequency.
    """
    nyquist = fs / 2.0

    fig, ax = plt.subplots(figsize=(10, 3))

    # Draw the frequency axis
    ax.axhline(0, color="black", linewidth=1.0)

    # Mark key frequencies
    freqs  = [0,        fc,           nyquist,        fs]
    labels = ["DC\n0 Hz",
              f"Fc\n{fc:.0f} Hz",
              f"Fs/2 (Nyquist)\n{nyquist:.0f} Hz",
              f"Fs\n{fs} Hz"]
    colors = ["black",  "C0",         "C3",           "gray"]

    for f, label, c in zip(freqs, labels, colors):
        ax.plot(f, 0, "o", color=c, markersize=10, zorder=5)
        ax.annotate(label, (f, 0), textcoords="offset points",
                    xytext=(0, 18), ha="center", fontsize=9, color=c,
                    fontweight="bold")

    # Shade the safe zone (0 to Fs/2)
    ax.axvspan(0, nyquist, alpha=0.08, color="green",
               label="Alias-free band")
    # Shade the danger zone (Fs/2 to Fs)
    ax.axvspan(nyquist, fs, alpha=0.08, color="red",
               label="Aliased / mirrored band")

    ax.set_xlim(-1000, fs + 2000)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("Module 2 -- Nyquist Sampling Theorem Compliance")
    ax.legend(loc="upper right")
    ax.set_yticks([])
    fig.tight_layout()

    path = os.path.join(output_dir, "nyquist_compliance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 2] Saved Nyquist diagram    -> {path}")
    return path


# MODULE RUNNER 

def run(bitstream, fs, fc, samples_per_bit, output_dir):
    """Execute all Module 2 tasks and return the modulated waveform.

    Parameters
    ----------
    bitstream : np.ndarray
        Binary sequence from Module 1.
    fs : int
        Sampling rate (Hz).
    fc : float
        Carrier frequency (Hz).
    samples_per_bit : int
        Samples per bit duration.
    output_dir : str
        Path for saving figures.

    Returns
    -------
    results : dict
        't':      time vector (seconds)
        'signal': BPSK modulated waveform array
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 2 -- Sampling & Continuous Waveform Synthesis")
    print("=" * 70)

    # Verify Nyquist compliance
    verify_nyquist(fc, fs)

    # Step 2: BPSK modulation 
    t, signal = bpsk_modulate(bitstream, fs, fc, samples_per_bit)

    bit_rate = fs / samples_per_bit
    duration_ms = len(signal) / fs * 1000

    print(f"\n  Modulation summary:")
    print(f"    Scheme:          BPSK (Binary Phase Shift Keying)")
    print(f"    Bit rate:        {bit_rate:.1f} bits/sec")
    print(f"    Samples/bit:     {samples_per_bit}")
    print(f"    Total samples:   {len(signal)}")
    print(f"    Waveform length: {duration_ms:.2f} ms")
    print(f"    Peak amplitude:  {np.max(np.abs(signal)):.4f}")

    # Generate plots
    plot_modulated_waveform(t, signal, fs, fc, samples_per_bit, bitstream,
                           output_dir)
    plot_nyquist_diagram(fc, fs, output_dir)

    return {
        "t":      t,
        "signal": signal,
    }
