
# Signal Identification & Array Initialization
#
# This module is the entry point of the transceiver.  It takes a plain
# ASCII string from the user and converts it into the fundamental
# discrete-time representation that every subsequent module operates on.


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving figures
import matplotlib.pyplot as plt


#CORE FUNCTIONS

def text_to_bitstream(text):
    """Convert an ASCII string into a 1-D numpy array of bits (0s and 1s).

    Each character is represented by 8 bits (MSB first), so the total
    sequence length is  N = 8 * len(text).

    Parameters
    ----------
    text : str
        The message to encode.  Only printable ASCII is expected.

    Returns
    -------
    x : np.ndarray, dtype int, shape (8 * len(text),)
        The discrete-time binary sequence x[n].

    Mathematical note
    -----------------
    x[n] is a finite-length (bounded support) sequence defined on
    the interval  0 <= n <= N-1,  where N = 8 * len(text).
    Outside this interval x[n] = 0 by convention.
    """
    bits = []
    for char in text:
        ascii_val = ord(char)
        # Extract 8 bits from MSB (bit 7) down to LSB (bit 0)
        for bit_pos in range(7, -1, -1):
            bits.append((ascii_val >> bit_pos) & 1)
    return np.array(bits, dtype=int)


def manual_convolution(x, h):
    """Compute the linear convolution sum of two finite-length sequences.

    This is the textbook definition implemented with explicit loops --
    no calls to np.convolve or scipy.signal are made.

    Definition (Discrete Convolution Sum):
        y[n] = sum_{k=0}^{M-1}  x[k] * h[n - k]

    where x has length L, h has length M, and the output y has length
    L + M - 1.

    Parameters
    ----------
    x : np.ndarray
        First input sequence of length L.
    h : np.ndarray
        Second input sequence (typically an impulse response) of length M.

    Returns
    -------
    y : np.ndarray of length L + M - 1
        The convolution result.
    """
    L = len(x)
    M = len(h)
    N = L + M - 1               # length of the full linear convolution
    y = np.zeros(N, dtype=float)

    # Walk through every output index n and accumulate the sum
    for n in range(N):
        for k in range(L):
            # h[n-k] is only valid when 0 <= n-k <= M-1
            if 0 <= (n - k) < M:
                y[n] += x[k] * h[n - k]
    return y


def classify_signal(x):
    """Classify a discrete-time sequence as an energy or power signal.

    Definitions used:
        Energy:   E = sum_{n=0}^{N-1} |x[n]|^2
        Power:    P = (1/N) * sum_{n=0}^{N-1} |x[n]|^2  =  E / N

    A signal with finite, non-zero energy is called an ENERGY SIGNAL.
    A signal with finite, non-zero average power (but infinite energy,
    e.g. a periodic signal) is called a POWER SIGNAL.

    For a finite-length bounded-support sequence like our bitstream,
    the energy is always finite, so it is an energy signal.

    Parameters
    ----------
    x : np.ndarray
        The discrete-time sequence to classify.

    Returns
    -------
    info : dict
        Keys: 'energy', 'power', 'length', 'classification', 'support'.
    """
    N = len(x)

    # Total energy -- sum of squared magnitudes
    energy = float(np.sum(np.abs(x) ** 2))

    # Average power -- energy normalised by the number of samples
    power = energy / N if N > 0 else 0.0

    # A bounded-support sequence with finite energy is an energy signal.
    # (An infinite-length periodic signal would be a power signal instead.)
    classification = "Energy Signal"

    info = {
        "energy":          energy,
        "power":           power,
        "length":          N,
        "classification":  classification,
        "support":         f"n in [0, {N - 1}]  (bounded / finite-length)",
    }
    return info


#PLOTTING

def plot_bitstream(x, output_dir):
    """Create a stem plot of the binary sequence x[n] and save to disk.

    The plot uses DSP conventions: the horizontal axis is the sample
    index n, and each value is shown as a vertical stem with a circle
    marker at the top.
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    n = np.arange(len(x))
    markerline, stemlines, baseline = ax.stem(n, x, linefmt="C0-",
                                               markerfmt="C0o",
                                               basefmt="k-")
    plt.setp(stemlines, linewidth=0.8)
    plt.setp(markerline, markersize=4)
    ax.set_xlabel("Sample index  n")
    ax.set_ylabel("x[n]")
    ax.set_title("Module 1 -- Binary Bitstream  x[n]")
    ax.set_yticks([0, 1])
    ax.set_xlim(-1, len(x))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "bitstream_stem.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 1] Saved bitstream stem plot -> {path}")
    return path


def plot_convolution_demo(x, h, y, output_dir):
    """Plot the two input sequences and their manual convolution result.

    This serves as a visual sanity check that our manual_convolution
    function produces the expected output shape and values.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=False)

    # Input sequence x[n]
    n_x = np.arange(len(x))
    axes[0].stem(n_x, x, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    axes[0].set_title("Input sequence  x[n]")
    axes[0].set_ylabel("x[n]")
    axes[0].grid(True, alpha=0.3)

    # Impulse response h[n]
    n_h = np.arange(len(h))
    axes[1].stem(n_h, h, linefmt="C1-", markerfmt="C1o", basefmt="k-")
    axes[1].set_title("Impulse response  h[n]")
    axes[1].set_ylabel("h[n]")
    axes[1].grid(True, alpha=0.3)

    # Convolution result y[n] = x[n] * h[n]
    n_y = np.arange(len(y))
    axes[2].stem(n_y, y, linefmt="C2-", markerfmt="C2o", basefmt="k-")
    axes[2].set_title("Convolution result  y[n] = x[n] * h[n]")
    axes[2].set_xlabel("Sample index  n")
    axes[2].set_ylabel("y[n]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "convolution_demo.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 1] Saved convolution demo   -> {path}")
    return path


#MODULE RUNNER

def run(text, output_dir):
    """Execute all Module 1 tasks and return results for the next stage.

    Parameters
    ----------
    text : str
        User-supplied ASCII message to encode.
    output_dir : str
        Directory where figures for this module will be saved.

    Returns
    -------
    results : dict
        'bitstream': np.ndarray of 0s and 1s
        'signal_info': dict from classify_signal
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 1 -- Signal Identification & Array Initialization")
    print("=" * 70)

    #Convert text to binary bitstream
    bitstream = text_to_bitstream(text)
    print(f"  Input text:      \"{text}\"")
    print(f"  Bitstream length: {len(bitstream)} bits  "
          f"({len(text)} chars x 8 bits)")
    # Show the first 24 bits for quick visual check
    preview = "".join(str(b) for b in bitstream[:24])
    print(f"  First 24 bits:   {preview}...")

    #Classify the signal
    info = classify_signal(bitstream)
    print(f"\n  Signal classification:")
    print(f"    Type:    {info['classification']}")
    print(f"    Support: {info['support']}")
    print(f"    Energy:  {info['energy']:.4f}")
    print(f"    Power:   {info['power']:.6f}")

    #Demonstrate manual convolutio
    # Use a simple 3-tap moving average as the test impulse response.
    # This is a basic FIR filter:  h[n] = [1/3, 1/3, 1/3]
    h_test = np.array([1.0/3, 1.0/3, 1.0/3])
    y_conv = manual_convolution(bitstream.astype(float), h_test)

    # Quick verification against numpy's own convolution
    y_check = np.convolve(bitstream.astype(float), h_test)
    max_error = np.max(np.abs(y_conv - y_check))
    print(f"\n  Manual convolution verification:")
    print(f"    h[n] = [1/3, 1/3, 1/3]  (3-tap moving average)")
    print(f"    Output length: {len(y_conv)}  (expected {len(bitstream) + len(h_test) - 1})")
    print(f"    Max error vs np.convolve: {max_error:.2e}")

    #  Generate plots 
    plot_bitstream(bitstream, output_dir)
    plot_convolution_demo(bitstream.astype(float), h_test, y_conv, output_dir)

    return {
        "bitstream":   bitstream,
        "signal_info": info,
    }
