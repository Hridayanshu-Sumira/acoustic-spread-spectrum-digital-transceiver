
#  Global System Constants
# Acoustic Spread-Spectrum Digital Transceiver
# Every module imports its physical and simulation parameters from here so
# there is exactly one place to tweak the system.  


import os

#  Sampling & Carrier
# Fs must be at least twice the highest frequency component (Nyquist).
# We use the standard CD-quality rate which is universally supported by
# audio hardware and gives plenty of headroom above our carrier.
FS = 44_100                     # Sampling rate in Hz

# The carrier sits comfortably in the mid-audio band.  BPSK modulation
# at 100 symbols/sec produces a main lobe of roughly Fc +/- 100 Hz,
# keeping all energy well below Fs/2 = 22 050 Hz.
FC = 3_000.0                    # Carrier frequency in Hz

# Symbol Timing 
SYMBOL_DURATION = 0.01          # Duration of one bit in seconds (10 ms)
SAMPLES_PER_BIT = int(FS * SYMBOL_DURATION)   # 441 samples per bit

# Channel Model 
# Simple multipath echo:  H(z) = 1 + alpha * z^(-N)
# One direct path plus one reflected path arriving N samples later.
ECHO_DELAY   = 50               # Delay of the reflected path in samples
ECHO_ALPHA   = 0.5              # Attenuation coefficient of the echo

# Noise
NOISE_SNR_DB = 15.0             # Signal-to-noise ratio for additive WGN

# IIR Bandpass Filter Design 
# We want to keep a band around the carrier and reject everything else.
# The passband is chosen wide enough to preserve the BPSK main lobe
# but narrow enough to reject most out-of-band noise.
IIR_ORDER     = 4               # Butterworth filter order (per side)
IIR_LOW_FREQ  = 2_000.0         # Lower passband edge in Hz
IIR_HIGH_FREQ = 4_000.0         # Upper passband edge in Hz

# FIR Low-Pass Filter Design (
FIR_NUM_TAPS   = 101            # Odd number keeps the filter symmetric
FIR_CUTOFF     = 200.0          # Cutoff frequency for baseband LPF in Hz
FIR_WINDOW     = "hamming"      # Window type for the FIR design

# Output Paths 
# Root output directory -- every module writes its plots here.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
