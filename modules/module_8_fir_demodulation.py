# ---------------------------------------------------------------------------
# module_8_fir_demodulation.py -- FIR Filter Windowing & Demodulation
#
# Demodulates the filtered BPSK signal to baseband, applies a custom FIR 
# low-pass filter (designed via Window Method) to reject the double-frequency 
# component, and performs bit detection to recover the original ASCII text.
# ---------------------------------------------------------------------------

import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def design_fir_lowpass(cutoff, fs, num_taps, window_type="hamming"):
    """Design an FIR low-pass filter using the Window Method."""
    # Compute ideal impulse response for LPF
    # h_ideal[n] = 2 * fc_norm * sinc(2 * fc_norm * (n - M))
    # where sinc(x) = sin(pi*x)/(pi*x) and fc_norm = cutoff / fs
    
    M = (num_taps - 1) / 2
    n = np.arange(num_taps)
    
    # Avoid division by zero at n = M
    with np.errstate(divide='ignore', invalid='ignore'):
        h_ideal = np.sin(2 * np.pi * (cutoff / fs) * (n - M)) / (np.pi * (n - M))
    h_ideal[int(M)] = 2 * (cutoff / fs)
    
    # Apply window function
    if window_type.lower() == "hamming":
        window = np.hamming(num_taps)
    elif window_type.lower() == "kaiser":
        window = np.kaiser(num_taps, beta=14)
    else:
        window = np.ones(num_taps)
        
    h = h_ideal * window
    return h


def fir_filter(h, x):
    """Apply an FIR filter using a manual convolution loop."""
    # Custom Direct Form FIR convolution
    # y[n] = sum_{k=0}^{M-1} h[k] * x[n - k]
    M = len(h)
    N = len(x)
    y = np.zeros(N)
    
    # We do a causal convolution matching the size of x
    for n in range(N):
        # Number of taps we can use at sample n
        k_max = min(n + 1, M)
        # y[n] = sum( h[0:k_max] * x[n : n-k_max : -1] )
        y[n] = np.sum(h[:k_max] * x[n::-1][:k_max])
        
    return y


def demodulate_bpsk(signal, fs, fc, samples_per_bit, num_bits, h_fir):
    """Demodulate BPSK signal back to binary stream using I/Q processing to handle arbitrary phase."""
    t = np.arange(len(signal)) / fs
    
    # 1. Multiply by local carrier (I and Q branches)
    mixed_I = signal * np.cos(2.0 * np.pi * fc * t)
    mixed_Q = signal * -np.sin(2.0 * np.pi * fc * t)
    
    # 2. Low-pass filter both branches to remove 2*Fc component
    baseband_I = fir_filter(h_fir, mixed_I)
    baseband_Q = fir_filter(h_fir, mixed_Q)
    
    # 3. Integrate over bit window
    i_symbols = np.zeros(num_bits)
    q_symbols = np.zeros(num_bits)
    
    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        
        i_symbols[i] = np.sum(baseband_I[start:end])
        q_symbols[i] = np.sum(baseband_Q[start:end])
        
    # 4. Phase synchronization
    # Combine I and Q into complex symbols
    complex_symbols = i_symbols + 1j * q_symbols
    
    # Squaring the BPSK symbols removes the 180 degree modulation (since (+1)^2 = (-1)^2 = 1)
    # The phase of the sum of squared symbols gives 2 * theta (where theta is the channel phase offset)
    squared_sum = np.sum(complex_symbols**2)
    phase_offset = 0.5 * np.angle(squared_sum)
    
    # Rotate symbols back to the real axis
    rotated_symbols = complex_symbols * np.exp(-1j * phase_offset)
    soft_bits = np.real(rotated_symbols)
    
    # Threshold to binary
    bits = (soft_bits > 0).astype(int)
    
    # 5. Resolve 180-degree ambiguity
    # In standard ASCII, the most significant bit (MSB) of characters is 0 (range 0-127).
    # We can check the first bit (MSB of the first char) to see if we are flipped.
    baseband = baseband_I * np.cos(-phase_offset) - baseband_Q * np.sin(-phase_offset)
    
    if len(bits) >= 8 and bits[0] == 1:
        bits = 1 - bits  # Flip all bits
        baseband = -baseband # Flip baseband plot as well
        
    return bits, mixed_I, baseband


def bitstream_to_text(bits):
    """Convert recovered binary array to ASCII text."""
    chars = []
    # Process bits in chunks of 8
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        # Convert array of 8 bits to integer
        val = 0
        for bit in byte:
            val = (val << 1) | bit
        chars.append(chr(val))
    return "".join(chars)


def plot_demodulation(t, mixed, baseband, samples_per_bit, fs, num_bits_show, output_dir):
    """Plot mixed and baseband signals for the first few bits."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    show_samples = min(num_bits_show * samples_per_bit, len(mixed))
    t_ms = t[:show_samples] * 1000
    
    axes[0].plot(t_ms, mixed[:show_samples], 'C0', linewidth=0.8)
    axes[0].set_title("Module 8 -- Multiplied with Local Carrier (Mixed)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_ms, baseband[:show_samples], 'C1', linewidth=1.5)
    axes[1].set_title("Module 8 -- Baseband Signal after FIR LPF")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    
    # Mark bit boundaries
    for b in range(num_bits_show + 1):
        boundary_ms = b * samples_per_bit / fs * 1000
        axes[1].axvline(boundary_ms, color="gray", linestyle="--", alpha=0.7)
        
    fig.tight_layout()
    path = os.path.join(output_dir, "demodulation_waveforms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 8] Saved demod waveforms    -> {path}")
    return path


def run(signal, fs, fc, samples_per_bit, num_bits, output_dir):
    """Execute Module 8: FIR design, demodulation, and text recovery."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 8 -- FIR Filter Windowing & Demodulation")
    print("=" * 70)

    # ---- Step 1: Design FIR Filter ----
    num_taps = 101
    cutoff = 200.0  # Hz
    print(f"  Designing FIR Low-Pass Filter (Window Method, {num_taps} taps, Fc={cutoff}Hz)")
    h_fir = design_fir_lowpass(cutoff, fs, num_taps, window_type="hamming")
    
    # ---- Step 2: Demodulate ----
    print(f"  Coherent BPSK demodulation & FIR filtering...")
    bits, mixed, baseband = demodulate_bpsk(signal, fs, fc, samples_per_bit, num_bits, h_fir)
    
    # ---- Step 3: Text Recovery ----
    recovered_text = bitstream_to_text(bits)
    print(f"\n  Recovered Text: \"{recovered_text}\"")
    
    # ---- Step 4: Plots ----
    t = np.arange(len(signal)) / fs
    plot_demodulation(t, mixed, baseband, samples_per_bit, fs, num_bits_show=min(4, num_bits // 8 * 8), output_dir=output_dir)

    return {
        "recovered_bits": bits,
        "decoded_text": recovered_text
    }
