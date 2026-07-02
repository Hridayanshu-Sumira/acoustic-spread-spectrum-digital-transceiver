
# Acoustic Transmission and Reception
#
# Handles playing the generated BPSK waveform over the computer's speakers
# and recording it via the microphone. Includes synchronization logic
# (preamble generation and cross-correlation detection) to align the
# recorded signal for the DSP demodulator.


import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import os

# A known barker-like sync sequence for robust cross-correlation
SYNC_BITS = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])

def modulate_sync_preamble(fs, fc, samples_per_bit):
    """Generate a BPSK modulated preamble."""
    t = np.arange(len(SYNC_BITS) * samples_per_bit) / fs
    sync_wave = np.zeros(len(t))
    
    for i, bit in enumerate(SYNC_BITS):
        start = i * samples_per_bit
        end = start + samples_per_bit
        phase = 0 if bit == 1 else np.pi
        sync_wave[start:end] = np.cos(2 * np.pi * fc * t[start:end] + phase)
        
    return sync_wave


def transmit_and_receive(tx_signal, fs, fc, samples_per_bit, record_padding=2.0):
    """
    Prepends a sync preamble to the signal, plays it out the speakers,
    and simultaneously records from the microphone.
    """
    #Prepend preamble
    preamble = modulate_sync_preamble(fs, fc, samples_per_bit)
    # Add some silence before and after
    silence = np.zeros(int(fs * 0.5))
    full_tx = np.concatenate([silence, preamble, tx_signal, silence])
    
    # Scale for audio playback
    full_tx = full_tx * 0.8 / np.max(np.abs(full_tx))
    
    print("  [Audio] Starting transmission and recording...")
    # Record for slightly longer than the transmission
    duration = len(full_tx) / fs + record_padding
    
    # We use sounddevice playrec to do both simultaneously
    # Play channel 1, record channel 1
    rec_data = sd.playrec(full_tx, samplerate=fs, channels=1, blocking=True)
    rx_signal = rec_data.flatten()
    print("  [Audio] Transmission complete.")
    
    return rx_signal, preamble


def synchronize_signal(rx_signal, preamble, tx_length, output_dir):
    """
    Find the preamble in the recorded signal using cross-correlation,
    and extract the actual payload.
    """
    print("  [Audio] Synchronizing received signal...")
    
    # We use numpy correlate (a standard block correlation, similar to matched filter)
    # to find where the preamble starts
    correlation = np.correlate(rx_signal, preamble, mode='valid')
    
    # Find the peak correlation
    peak_idx = np.argmax(np.abs(correlation))
    
    # The payload starts immediately after the preamble
    payload_start = peak_idx + len(preamble)
    payload_end = payload_start + tx_length
    
    # Plot correlation for debugging
    plt.figure(figsize=(10, 4))
    plt.plot(correlation)
    plt.axvline(peak_idx, color='r', linestyle='--', label='Detected Sync Peak')
    plt.title("Audio Sync Cross-Correlation")
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sync_correlation.png"))
    plt.close()
    
    # Extract just the payload
    # Add a small safeguard if recording cut off early
    if payload_end > len(rx_signal):
        print("  [Audio WARNING] Recording cut off early! Padding with zeros.")
        padding = np.zeros(payload_end - len(rx_signal))
        rx_signal = np.concatenate([rx_signal, padding])
        
    extracted_payload = rx_signal[payload_start:payload_end]
    return extracted_payload
