import os
import sys
import threading
import numpy as np
import customtkinter as ctk
from PIL import Image
import sounddevice as sd

import main
import config

# Set overall appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.after(0, self._insert_text, string)
        
    def _insert_text(self, string):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", string)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass

class ModernTransceiverUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Acoustic Spread-Spectrum Digital Transceiver")
        self.geometry("1300x850")
        
        # Configure fonts
        self.font_title = ctk.CTkFont(family="Segoe UI", size=26, weight="bold")
        self.font_subtitle = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        self.font_normal = ctk.CTkFont(family="Segoe UI", size=14)
        self.font_terminal = ctk.CTkFont(family="Consolas", size=14)

        self.pipeline = main.TransceiverPipeline()

        # Grid Layout
        self.grid_columnconfigure(1, weight=1) # Main content area
        self.grid_rowconfigure(0, weight=3)    # Dashboard (top)
        self.grid_rowconfigure(1, weight=2)    # Terminal (bottom)

        self.create_sidebar()
        self.create_main_view()
        self.create_terminal_view()
        
        self.image_references = {} 

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#1a1a1a")
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Title
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.grid(row=0, column=0, pady=(30, 20), sticky="ew")
        ctk.CTkLabel(title_frame, text="DSP Modem", font=self.font_title, text_color="#3498db").pack()
        ctk.CTkLabel(title_frame, text="Software Defined Acoustic Transceiver", font=ctk.CTkFont(size=11), text_color="gray").pack()

        # --- Section 1: Input ---
        input_frame = ctk.CTkFrame(self.sidebar, fg_color="#2a2a2a", corner_radius=10)
        input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        ctk.CTkLabel(input_frame, text="Data Payload", font=self.font_subtitle).pack(pady=(10, 5))
        
        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Enter Message...", font=self.font_normal, height=35)
        self.input_entry.insert(0, "KU")
        self.input_entry.pack(padx=15, pady=(0, 15), fill="x")

        # --- Section 2: Pipeline Controls ---
        control_frame = ctk.CTkFrame(self.sidebar, fg_color="#2a2a2a", corner_radius=10)
        control_frame.grid(row=2, column=0, padx=20, pady=0, sticky="ew")
        ctk.CTkLabel(control_frame, text="Pipeline Controls", font=self.font_subtitle).pack(pady=(10, 5))

        self.btn_step1 = ctk.CTkButton(control_frame, text="1. TX Encode & Modulate", font=self.font_normal, height=35, command=self.run_step1)
        self.btn_step1.pack(padx=15, pady=5, fill="x")

        self.btn_step2a = ctk.CTkButton(control_frame, text="2a. Channel (Simulated)", font=self.font_normal, height=35, fg_color="#555555", hover_color="#666666", command=self.run_step2a)
        self.btn_step2a.pack(padx=15, pady=5, fill="x")

        self.btn_step2b = ctk.CTkButton(control_frame, text="2b. Channel (Acoustic)", font=self.font_normal, height=35, fg_color="#d35400", hover_color="#e67e22", command=self.run_step2b)
        self.btn_step2b.pack(padx=15, pady=5, fill="x")

        self.btn_step3 = ctk.CTkButton(control_frame, text="3. RX Filter & Decode", font=self.font_normal, height=35, fg_color="#555555", hover_color="#666666", command=self.run_step3)
        self.btn_step3.pack(padx=15, pady=(5, 15), fill="x")

        # --- Section 3: Audio Playback ---
        self.audio_frame = ctk.CTkFrame(self.sidebar, fg_color="#2a2a2a", corner_radius=10)
        self.audio_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        ctk.CTkLabel(self.audio_frame, text="Signal Inspection", font=self.font_subtitle).pack(pady=(10, 5))
        
        self.btn_play_tx = ctk.CTkButton(self.audio_frame, text="🔊 Play TX Signal", font=self.font_normal, height=30, fg_color="#27ae60", hover_color="#2ecc71", state="disabled", command=self.play_tx)
        self.btn_play_tx.pack(padx=15, pady=5, fill="x")
        
        self.btn_play_rx = ctk.CTkButton(self.audio_frame, text="🔊 Play RX Signal", font=self.font_normal, height=30, fg_color="#8e44ad", hover_color="#9b59b6", state="disabled", command=self.play_rx)
        self.btn_play_rx.pack(padx=15, pady=(5, 15), fill="x")

        # --- Output Result ---
        self.result_frame = ctk.CTkFrame(self.sidebar, fg_color="#1e3a2a", border_width=1, border_color="#2ecc71", corner_radius=10)
        self.result_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.output_label = ctk.CTkLabel(self.result_frame, text="Decoded Output:\nWAITING...", font=self.font_subtitle, text_color="#a8e6cf")
        self.output_label.pack(pady=15)

        # Bottom Controls
        self.btn_clear = ctk.CTkButton(self.sidebar, text="Reset Pipeline", font=self.font_normal, height=35, fg_color="#c0392b", hover_color="#e74c3c", command=self.clear_results)
        self.btn_clear.grid(row=8, column=0, padx=20, pady=10, sticky="ew")

        # Status Bar & Progress
        self.progress = ctk.CTkProgressBar(self.sidebar, mode="indeterminate", height=6)
        self.progress.grid(row=9, column=0, padx=20, pady=(0, 5), sticky="ew")
        self.progress.set(0)
        
        self.status_var = ctk.StringVar(value="System Ready")
        self.status_label = ctk.CTkLabel(self.sidebar, textvariable=self.status_var, font=ctk.CTkFont(size=12, weight="bold"), text_color="#95a5a6")
        self.status_label.grid(row=10, column=0, padx=20, pady=(0, 20), sticky="s")

    def create_main_view(self):
        self.main_view = ctk.CTkFrame(self, corner_radius=15, fg_color="#202020")
        self.main_view.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="nsew")
        self.main_view.grid_rowconfigure(1, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1)
        
        # Header
        header = ctk.CTkFrame(self.main_view, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=15)
        ctk.CTkLabel(header, text="Visual Analysis Dashboard", font=self.font_title).pack(side="left")
        
        # Modern TabView
        self.tabview = ctk.CTkTabview(self.main_view, corner_radius=10, fg_color="#2a2a2a", segmented_button_selected_color="#3498db", segmented_button_selected_hover_color="#2980b9")
        self.tabview.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        self.tabs = [
            "Module 1: Baseband",
            "Module 2: Modulation",
            "Module 3-4: Spectrum",
            "Module 5: Channel",
            "Module 6: DF2T Filter",
            "Module 7: IIR Design",
            "Module 8: Demodulation",
            "Audio Sync"
        ]
        self.image_frames = {}
        for tab_name in self.tabs:
            self.tabview.add(tab_name)
            
            scroll_frame = ctk.CTkScrollableFrame(self.tabview.tab(tab_name), fg_color="transparent")
            scroll_frame.pack(expand=True, fill="both")
            self.image_frames[tab_name] = scroll_frame
            
            lbl = ctk.CTkLabel(scroll_frame, text="Run the pipeline to generate visual data.", font=self.font_normal, text_color="gray")
            lbl.pack(expand=True, fill="both", pady=50)

    def create_terminal_view(self):
        self.term_view = ctk.CTkFrame(self, corner_radius=15, fg_color="#181818")
        self.term_view.grid(row=1, column=1, padx=20, pady=(10, 20), sticky="nsew")
        self.term_view.grid_rowconfigure(1, weight=1)
        self.term_view.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(self.term_view, text="DSP Process Log", font=self.font_subtitle, text_color="#3498db")
        title.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")
        
        self.console_textbox = ctk.CTkTextbox(self.term_view, font=self.font_terminal, fg_color="#0c0c0c", text_color="#00ff00", corner_radius=8)
        self.console_textbox.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.console_textbox.configure(state="disabled")
        
        # Redirect stdout
        sys.stdout = StdoutRedirector(self.console_textbox)

    def set_status(self, text, is_working=False):
        self.status_var.set(text)
        if is_working:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)
        self.update()

    def clear_results(self):
        # Reset visual state
        self.btn_step2a.configure(fg_color="#555555")
        self.btn_step2b.configure(fg_color="#d35400")
        self.btn_step3.configure(fg_color="#555555")
        self.btn_play_tx.configure(state="disabled")
        self.btn_play_rx.configure(state="disabled")
        
        self.result_frame.configure(border_color="#2ecc71")
        self.output_label.configure(text="Decoded Output:\nWAITING...", text_color="#a8e6cf")
        
        self.console_textbox.configure(state="normal")
        self.console_textbox.delete("0.0", "end")
        self.console_textbox.configure(state="disabled")
        
        for tab_name in self.tabs:
            for widget in self.image_frames[tab_name].winfo_children():
                widget.destroy()
            lbl = ctk.CTkLabel(self.image_frames[tab_name], text="Run the pipeline to generate visual data.", font=self.font_normal, text_color="gray")
            lbl.pack(expand=True, fill="both", pady=50)
        self.image_references = {}
        
        self.pipeline = main.TransceiverPipeline()
        self.set_status("System Reset")

    def run_step1(self):
        txt = self.input_entry.get().strip()
        if not txt:
            return
            
        def task():
            self.after(0, lambda: self.set_status("Encoding & Modulating...", True))
            self.pipeline.encode_and_modulate(txt)
            
            self.after(0, lambda: self.load_images_into_tab("Module 1: Baseband", [
                os.path.join(config.OUTPUT_DIR, "module_1", "bitstream_stem.png"),
                os.path.join(config.OUTPUT_DIR, "module_1", "convolution_demo.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab("Module 2: Modulation", [
                os.path.join(config.OUTPUT_DIR, "module_2", "bpsk_waveform_zoomed.png"),
                os.path.join(config.OUTPUT_DIR, "module_2", "bpsk_waveform_full.png"),
                os.path.join(config.OUTPUT_DIR, "module_2", "nyquist_compliance.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab("Module 3-4: Spectrum", [
                os.path.join(config.OUTPUT_DIR, "module_3_4", "fft_magnitude_spectrum.png")
            ]))
            
            self.after(0, lambda: self.btn_step2a.configure(fg_color=["#3B8ED0", "#1F6AA5"]))
            self.after(0, lambda: self.btn_play_tx.configure(state="normal"))
            self.after(0, lambda: self.set_status("Step 1 Complete. Awaiting Channel.", False))

        threading.Thread(target=task, daemon=True).start()

    def run_step2a(self):
        if self.pipeline.tx_signal is None:
            return
            
        def task():
            self.after(0, lambda: self.set_status("Simulating Multipath & AWGN...", True))
            self.pipeline.run_channel_simulation()
            
            self.after(0, lambda: self.load_images_into_tab("Module 5: Channel", [
                os.path.join(config.OUTPUT_DIR, "module_5", "channel_effect.png"),
                os.path.join(config.OUTPUT_DIR, "module_5", "pole_zero_map.png"),
                os.path.join(config.OUTPUT_DIR, "module_5", "channel_impulse_response.png")
            ]))
            self.after(0, lambda: self.tabview.set("Module 5: Channel"))
            
            self.after(0, lambda: self.btn_step3.configure(fg_color=["#3B8ED0", "#1F6AA5"]))
            self.after(0, lambda: self.btn_play_rx.configure(state="normal"))
            self.after(0, lambda: self.set_status("Simulation Complete. Awaiting Decode.", False))

        threading.Thread(target=task, daemon=True).start()

    def run_step2b(self):
        if self.pipeline.tx_signal is None:
            return
            
        def task():
            self.after(0, lambda: self.set_status("Acoustic TX/RX Active...", True))
            try:
                self.pipeline.run_channel_acoustic()
                
                self.after(0, lambda: self.load_images_into_tab("Audio Sync", [
                    os.path.join(config.OUTPUT_DIR, "module_sync", "sync_correlation.png")
                ]))
                self.after(0, lambda: self.tabview.set("Audio Sync"))
                
                self.after(0, lambda: self.btn_step3.configure(fg_color=["#3B8ED0", "#1F6AA5"]))
                self.after(0, lambda: self.btn_play_rx.configure(state="normal"))
                self.after(0, lambda: self.set_status("Acoustic Transfer Complete. Awaiting Decode.", False))
            except Exception as e:
                print(f"Audio Error: {e}")
                self.after(0, lambda: self.set_status(f"Error: {str(e)}", False))

        threading.Thread(target=task, daemon=True).start()

    def run_step3(self):
        if self.pipeline.rx_signal is None:
            return
            
        def task():
            self.after(0, lambda: self.set_status("Filtering & Demodulating...", True))
            recovered = self.pipeline.decode_and_demodulate()
            
            self.after(0, lambda: self.load_images_into_tab("Module 6: DF2T Filter", [
                os.path.join(config.OUTPUT_DIR, "module_6", "df2t_impulse_response.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab("Module 7: IIR Design", [
                os.path.join(config.OUTPUT_DIR, "module_7", "iir_frequency_response.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab("Module 8: Demodulation", [
                os.path.join(config.OUTPUT_DIR, "module_8", "demodulation_waveforms.png")
            ]))
            self.after(0, lambda: self.tabview.set("Module 8: Demodulation"))
            
            if recovered == self.pipeline.input_text:
                self.after(0, lambda: self.result_frame.configure(border_color="#2ecc71"))
                self.after(0, lambda: self.output_label.configure(text=f"Decoded Output:\n{recovered}", text_color="#2ecc71"))
            else:
                self.after(0, lambda: self.result_frame.configure(border_color="#e74c3c"))
                self.after(0, lambda: self.output_label.configure(text=f"Decoded Output:\n{recovered}", text_color="#e74c3c"))
                
            self.after(0, lambda: self.set_status("Pipeline Complete!", False))

        threading.Thread(target=task, daemon=True).start()

    def load_images_into_tab(self, tab_name, filepaths):
        # Clear existing widgets in the scrollable frame
        for widget in self.image_frames[tab_name].winfo_children():
            widget.destroy()
            
        for filepath in filepaths:
            if os.path.exists(filepath):
                img = Image.open(filepath)
                # Resize image dynamically to fit tab
                w, h = img.size
                ratio = min(750/w, 500/h)
                new_w, new_h = int(w*ratio), int(h*ratio)
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
                
                if tab_name not in self.image_references:
                    self.image_references[tab_name] = []
                self.image_references[tab_name].append(ctk_img)
                
                lbl = ctk.CTkLabel(self.image_frames[tab_name], image=ctk_img, text="")
                lbl.pack(pady=10)

    def play_tx(self):
        if self.pipeline.tx_signal is not None:
            sig = self.pipeline.tx_signal * 0.8 / np.max(np.abs(self.pipeline.tx_signal))
            sd.play(sig, self.pipeline.fs)

    def play_rx(self):
        if self.pipeline.rx_signal is not None:
            sig = self.pipeline.rx_signal * 0.8 / np.max(np.abs(self.pipeline.rx_signal))
            sd.play(sig, self.pipeline.fs)

if __name__ == "__main__":
    app = ModernTransceiverUI()
    app.mainloop()
