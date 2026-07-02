
# Z-Transform Channel Modeling
#
# Models the acoustic transmission channel as a discrete LTI system that
# introduces multipath echo reflections and additive white Gaussian noise.


# Channel Model:
#   H(z) = 1 + alpha * z^{-N}

# This represents a direct path (gain = 1) plus a single reflected path
# arriving N samples later with attenuation alpha.

# Difference equation:
#   y[n] = x[n] + alpha * x[n - N]

# Since the system is all-zeros (FIR), all poles lie at z = 0, which is
# inside the unit circle.  The system is therefore always BIBO stable.


import os
import numpy as np
import matplotlib.pyplot as plt


# CORE FUNCTIONS

def create_channel_impulse_response(alpha, delay):
    """Build the impulse response h[n] for the multipath echo channel.

    H(z) = 1 + alpha * z^{-N}  corresponds to:

        h[n] = delta[n] + alpha * delta[n - N]

    where delta[n] is the unit impulse (Kronecker delta).

    Parameters
    ----------
    alpha : float
        Echo attenuation coefficient.
    delay : int
        Echo delay in samples (N).

    Returns
    -------
    h : np.ndarray
        Impulse response array of length delay + 1.
    """
    h = np.zeros(delay + 1)
    h[0] = 1.0           # direct path
    h[delay] = alpha      # reflected path
    return h


def apply_channel(signal, alpha, delay):
    """Pass a signal through the multipath echo channel.

    Implements the difference equation directly using numpy vector
    operations (no external filtering library):

        y[n] = x[n] + alpha * x[n - N]

    Parameters
    ----------
    signal : np.ndarray
        Clean input waveform.
    alpha : float
        Echo attenuation.
    delay : int
        Echo delay in samples.

    Returns
    -------
    output : np.ndarray
        Channel-corrupted signal (same length as input).
    """
    output = signal.copy()
    # Add the delayed, attenuated echo.  For n < delay, x[n-delay] = 0
    # by the causal assumption, so we only modify samples from index
    # 'delay' onward.
    output[delay:] += alpha * signal[:-delay]
    return output


def add_noise(signal, snr_db, seed=42):
    """Add white Gaussian noise to achieve a target signal-to-noise ratio.

    SNR is defined as:
        SNR_dB = 10 * log10( P_signal / P_noise )

    so:
        P_noise = P_signal / 10^(SNR_dB / 10)

    Parameters
    ----------
    signal : np.ndarray
        Clean (or channel-corrupted) signal.
    snr_db : float
        Desired SNR in decibels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    noisy : np.ndarray
        Signal with additive white Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power) * rng.standard_normal(len(signal))
    return signal + noise


def find_poles_zeros(b, a):
    """Extract poles and zeros from a discrete system H(z) = B(z)/A(z).

    The zeros are the roots of the numerator polynomial B(z).
    The poles are the roots of the denominator polynomial A(z).

    For an FIR system (a = [1]), all poles are at z = 0.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.

    Returns
    -------
    zeros : np.ndarray
        Complex zeros of H(z).
    poles : np.ndarray
        Complex poles of H(z).
    """
    zeros = np.roots(b)
    poles = np.roots(a) if len(a) > 1 else np.array([])
    return zeros, poles


# ============================= PLOTTING ====================================

def plot_channel_impulse(h, output_dir):
    """Stem plot of the channel impulse response h[n]."""
    fig, ax = plt.subplots(figsize=(12, 3))
    n = np.arange(len(h))
    ax.stem(n, h, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax.set_xlabel("Sample index  n")
    ax.set_ylabel("h[n]")
    ax.set_title("Module 5 -- Channel Impulse Response  h[n]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "channel_impulse_response.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 5] Saved impulse response   -> {path}")
    return path


def plot_channel_effect(t, clean, corrupted, output_dir, samples_to_show=2000):
    """Compare the clean signal and channel-corrupted signal in time domain."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    sl = slice(0, samples_to_show)
    t_ms = t[sl] * 1000

    axes[0].plot(t_ms, clean[sl], linewidth=0.5, color="C0")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Module 5 -- Clean Transmitted Signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, corrupted[sl], linewidth=0.5, color="C1")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Module 5 -- After Channel (echo + noise)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "channel_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 5] Saved channel effect     -> {path}")
    return path


def plot_pole_zero(zeros, poles, alpha, delay, output_dir):
    """Plot the pole-zero constellation on the Z-plane.

    The unit circle is drawn for reference.  If all poles lie strictly
    inside the unit circle, the system is BIBO stable.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw the unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.4,
            label="Unit circle")

    # Plot zeros
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros), "bo", markersize=6,
                label=f"Zeros ({len(zeros)})", alpha=0.7)
    # Plot poles
    if len(poles) > 0:
        ax.plot(np.real(poles), np.imag(poles), "rx", markersize=8,
                markeredgewidth=2, label=f"Poles ({len(poles)})")
    else:
        # FIR system: all poles at origin
        ax.plot(0, 0, "rx", markersize=10, markeredgewidth=2,
                label=f"Poles ({delay} at origin)")

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(f"Module 5 -- Pole-Zero Map  (alpha={alpha}, delay={delay})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Annotate stability
    ax.text(0.02, 0.02,
            "All poles at z=0 (inside unit circle)\n-> System is BIBO stable",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.6))

    fig.tight_layout()
    path = os.path.join(output_dir, "pole_zero_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 5] Saved pole-zero map      -> {path}")
    return path


# MODULE RUNNER 

def run(signal, t, alpha, delay, snr_db, output_dir):
    """Execute Module 5: channel modeling and Z-plane analysis.

    Parameters
    ----------
    signal : np.ndarray
        Clean BPSK waveform from Module 2.
    t : np.ndarray
        Time vector corresponding to the signal.
    alpha : float
        Echo attenuation coefficient.
    delay : int
        Echo delay in samples.
    snr_db : float
        Target signal-to-noise ratio in dB.
    output_dir : str
        Directory for output figures.

    Returns
    -------
    results : dict
        'corrupted_signal': signal after channel + noise
        'channel_h':        impulse response
        'zeros', 'poles':   system zeros and poles
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 5 -- Z-Transform Channel Modeling")
    print("=" * 70)

    # Create channel impulse response 
    h = create_channel_impulse_response(alpha, delay)
    print(f"  Channel model: H(z) = 1 + {alpha} * z^(-{delay})")
    print(f"  Impulse response length: {len(h)} samples")

    # Apply channel (echo) 
    echoed = apply_channel(signal, alpha, delay)
    print(f"  Applied multipath echo to signal")

    # Add noise 
    corrupted = add_noise(echoed, snr_db)
    noise_power = np.mean((corrupted - echoed) ** 2)
    sig_power = np.mean(echoed ** 2)
    measured_snr = 10 * np.log10(sig_power / noise_power)
    print(f"  Added AWGN at target SNR = {snr_db:.1f} dB")
    print(f"  Measured SNR = {measured_snr:.1f} dB")

    # Find poles and zeros 
    b_coeffs = h.copy()                        # numerator = h[n]
    a_coeffs = np.array([1.0])                 # denominator = 1 (FIR)
    # For a proper pole representation, include the z^N denominator
    a_full = np.zeros(delay + 1)
    a_full[0] = 1.0
    zeros, poles = find_poles_zeros(b_coeffs, a_full)

    print(f"\n  Z-plane analysis:")
    print(f"    Number of zeros: {len(zeros)}")
    print(f"    Number of poles: {len(poles)}")
    print(f"    Max pole magnitude: {np.max(np.abs(poles)) if len(poles) > 0 else 0:.6f}")
    print(f"    Stability: BIBO stable (all poles at origin)")

    # Generate plots 
    plot_channel_impulse(h, output_dir)
    plot_channel_effect(t, signal, corrupted, output_dir)
    plot_pole_zero(zeros, poles, alpha, delay, output_dir)

    return {
        "corrupted_signal": corrupted,
        "channel_h":        h,
        "zeros":            zeros,
        "poles":            poles,
        "b_coeffs":         b_coeffs,
        "a_coeffs":         a_coeffs,
    }
