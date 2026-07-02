

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
        config.OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", safe_text)
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
        """Step 2b: Physically transmit and receive via speakers/mic."""
        print("\n=== CHANNEL (ACOUSTIC) ===")
        raw_rx, preamble = module_audio_io.transmit_and_receive(
            self.tx_signal, self.fs, self.fc, config.SAMPLES_PER_BIT
        )
        
        # Synchronize
        self.rx_signal = module_audio_io.synchronize_signal(
            raw_rx, preamble, len(self.tx_signal), 
            output_dir=os.path.join(config.OUTPUT_DIR, "module_sync")
        )
        return self.rx_signal

    def decode_and_demodulate(self):
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
        num_bits = len(self.bitstream)
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
