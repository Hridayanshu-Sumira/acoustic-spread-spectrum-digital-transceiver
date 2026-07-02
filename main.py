

import os
import sys
import numpy as np

#Project imports
import config
from modules import module_1_signals
from modules import module_2_sampling
from modules import module_3_4_fourier
from modules import module_5_ztransform
from modules import module_6_filter_structures
from modules import module_7_iir_design
from modules import module_8_fir_demodulation
from modules import module_audio_io


class TransceiverPipeline:
    def __init__(self):
        self.input_text = ""
        self.bitstream = None
        self.tx_signal = None
        self.rx_signal = None
        self.recovered_text = ""
        self.fs = config.FS
        self.fc = config.FC

    def encode_and_modulate(self, input_text):
        """Step 1: Convert text to BPSK waveform."""
        self.input_text = input_text
        
        # Create a safe folder name from input text
        safe_text = "".join([c if c.isalnum() else "_" for c in input_text])[:20]
        config.OUTPUT_DIR = os.path.join(config.get_base_dir(), "output", safe_text)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        print("\n=== TRANSMITTER ===")
        
        # Module 1
        mod1_out = module_1_signals.run(
            text=input_text,
            output_dir=os.path.join(config.OUTPUT_DIR, "module_1"),
        )
        self.bitstream = mod1_out["bitstream"]

        # Module 2
        mod2_out = module_2_sampling.run(
            bitstream=self.bitstream,
            fs=self.fs,
            fc=self.fc,
            samples_per_bit=config.SAMPLES_PER_BIT,
            output_dir=os.path.join(config.OUTPUT_DIR, "module_2"),
        )
        self.tx_signal = mod2_out["signal"]
        
        # Module 3-4 (Verify spectrum)
        module_3_4_fourier.run(
            signal=self.tx_signal,
            fs=self.fs,
            fc=self.fc,
            output_dir=os.path.join(config.OUTPUT_DIR, "module_3_4"),
        )
        return self.tx_signal

    def run_channel_simulation(self):
        """Step 2a: Run the signal through simulated multipath and noise."""
        print("\n=== CHANNEL (SIMULATED) ===")
        t_signal = np.arange(len(self.tx_signal)) / self.fs
        mod5_out = module_5_ztransform.run(
            signal=self.tx_signal,
            t=t_signal,
            alpha=config.ECHO_ALPHA,
            delay=config.ECHO_DELAY,
            snr_db=config.NOISE_SNR_DB,
            output_dir=os.path.join(config.OUTPUT_DIR, "module_5"),
        )
        self.rx_signal = mod5_out["corrupted_signal"]
        return self.rx_signal

    def run_channel_acoustic(self):
        """Step 2b: Physically transmit and receive via speakers/mic (same device)."""
        print("\n=== CHANNEL (ACOUSTIC — PLAY + RECORD) ===")
        raw_rx, preamble = module_audio_io.transmit_and_receive(
            self.tx_signal, self.fs, self.fc, config.SAMPLES_PER_BIT
        )
        
        # Synchronize
        self.rx_signal = module_audio_io.synchronize_signal(
            raw_rx, preamble, len(self.tx_signal), 
            output_dir=os.path.join(config.OUTPUT_DIR, "module_sync")
        )
        return self.rx_signal

    def run_record_only(self):
        """Step 2b (alt): Record from mic only — TX signal played on external device."""
        print("\n=== CHANNEL (ACOUSTIC — RECORD ONLY) ===")
        raw_rx, preamble = module_audio_io.record_only(
            len(self.tx_signal), self.fs, self.fc, config.SAMPLES_PER_BIT
        )

        # Synchronize
        self.rx_signal = module_audio_io.synchronize_signal(
            raw_rx, preamble, len(self.tx_signal),
            output_dir=os.path.join(config.OUTPUT_DIR, "module_sync")
        )
        return self.rx_signal

    def decode_and_demodulate(self, num_bits_override=None):
        """Step 3: Filter out noise and demodulate BPSK to text."""
        print("\n=== RECEIVER ===")
        
        # Module 6 (Verify DF2T logic)
        module_6_filter_structures.run(
            output_dir=os.path.join(config.OUTPUT_DIR, "module_6")
        )

        # Module 7 (IIR Bandpass)
        mod7_out = module_7_iir_design.run(
            signal=self.rx_signal,
            order=config.IIR_ORDER,
            f_low=config.IIR_LOW_FREQ,
            f_high=config.IIR_HIGH_FREQ,
            fs=self.fs,
            output_dir=os.path.join(config.OUTPUT_DIR, "module_7"),
        )
        filtered_rx = mod7_out["filtered_signal"]

        # Module 8 (FIR Demodulation)
        if num_bits_override is not None:
            num_bits = num_bits_override
        elif self.bitstream is not None:
            num_bits = len(self.bitstream)
        else:
            # Auto-detect: derive from rx_signal length
            num_bits = (len(self.rx_signal) // config.SAMPLES_PER_BIT // 8) * 8
            print(f"  [RX] Auto-detected num_bits = {num_bits} ({num_bits // 8} chars)")

        mod8_out = module_8_fir_demodulation.run(
            signal=filtered_rx,
            fs=self.fs,
            fc=self.fc,
            samples_per_bit=config.SAMPLES_PER_BIT,
            num_bits=num_bits,
            output_dir=os.path.join(config.OUTPUT_DIR, "module_8"),
        )
        
        self.recovered_text = mod8_out["decoded_text"]
        return self.recovered_text

    def receive_only(self, duration_s=10.0, num_chars=0):
        """
        Standalone receive: record from mic and decode without any prior TX step.
        The sender must have used the same system (preamble + BPSK payload).
        
        Args:
            duration_s: How long to record in seconds.
            num_chars:  Expected number of characters. 0 = auto-detect from signal length.
        """
        import sounddevice as sd
        import matplotlib.pyplot as plt

        # Create output dir (no input_text known, use 'rx_only')
        config.OUTPUT_DIR = os.path.join(
            config.get_base_dir(), "output", "rx_only"
        )
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        print(f"\n=== STANDALONE RECEIVER (recording {duration_s:.1f}s) ===")
        print("  [RX] Microphone is open — play the TX audio on the other device now!")
        rec_data = sd.rec(int(duration_s * self.fs), samplerate=self.fs,
                          channels=1, blocking=True)
        raw_rx = rec_data.flatten()
        print("  [RX] Recording complete. Searching for preamble...")

        # Build the same preamble the sender prepended
        preamble = module_audio_io.modulate_sync_preamble(
            self.fs, self.fc, config.SAMPLES_PER_BIT
        )

        # Cross-correlate to find sync peak
        correlation = np.correlate(raw_rx, preamble, mode='valid')
        peak_idx = int(np.argmax(np.abs(correlation)))
        payload_start = peak_idx + len(preamble)
        print(f"  [RX] Preamble detected at sample {peak_idx} "
              f"({peak_idx / self.fs:.3f}s into recording)")

        # Determine payload length
        if num_chars > 0:
            num_bits = num_chars * 8
            payload_end = payload_start + num_bits * config.SAMPLES_PER_BIT
        else:
            # Take everything after the preamble (up to end of recording)
            remaining = len(raw_rx) - payload_start
            num_bits = (remaining // config.SAMPLES_PER_BIT // 8) * 8
            payload_end = payload_start + num_bits * config.SAMPLES_PER_BIT
            print(f"  [RX] Auto-detected {num_bits} bits ({num_bits // 8} chars)")

        # Guard against recording cut-off
        if payload_end > len(raw_rx):
            padding = np.zeros(payload_end - len(raw_rx))
            raw_rx = np.concatenate([raw_rx, padding])
            print("  [RX WARNING] Recording too short — padded with zeros.")

        self.rx_signal = raw_rx[payload_start:payload_end]
        # Reset bitstream so decode_and_demodulate uses auto num_bits path
        self.bitstream = None

        # Save sync plot
        sync_dir = os.path.join(config.OUTPUT_DIR, "module_sync")
        os.makedirs(sync_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(correlation)
        plt.axvline(peak_idx, color='r', linestyle='--', label='Detected Sync Peak')
        plt.title("Standalone RX — Audio Sync Cross-Correlation")
        plt.xlabel("Sample Index")
        plt.ylabel("Correlation Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sync_dir, "sync_correlation.png"))
        plt.close()

        recovered_text = self.decode_and_demodulate(num_bits_override=num_bits)

        # Rename the output directory to match the decoded text
        import shutil
        safe_text = "".join([c if c.isalnum() else "_" for c in recovered_text])[:20]
        if not safe_text:
            safe_text = "rx_empty"
            
        final_dir = os.path.join(os.path.dirname(config.OUTPUT_DIR), f"rx_{safe_text}")
        
        try:
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.rename(config.OUTPUT_DIR, final_dir)
            config.OUTPUT_DIR = final_dir
        except Exception as e:
            print(f"  [RX WARNING] Could not rename output dir: {e}")

        return recovered_text




def main():
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = input("Enter message to transmit (e.g. KU): ").strip()

    if not input_text:
        return

    mode = input("Use acoustic transmission over speakers? (y/N): ").strip().lower()

    pipeline = TransceiverPipeline()
    pipeline.encode_and_modulate(input_text)
    
    import numpy as np # Ensure numpy is available for t_signal in simulation
    
    if mode == 'y':
        pipeline.run_channel_acoustic()
    else:
        pipeline.run_channel_simulation()
        
    recovered = pipeline.decode_and_demodulate()
    print("\n" + "=" * 50)
    print(f"Original:  {input_text}")
    print(f"Recovered: {recovered}")
    print("=" * 50)


if __name__ == "__main__":
    main()
