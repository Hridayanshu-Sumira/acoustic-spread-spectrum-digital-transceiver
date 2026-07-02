import os
import sys
import threading
import tkinter as tk
import numpy as np
import customtkinter as ctk
from PIL import Image
import sounddevice as sd
import scipy.io.wavfile as wavfile
import tkinter.filedialog as filedialog

import main
import config

# Design system

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg":           "#0f1115",   # app background
    "sidebar":      "#14161c",   # sidebar background
    "surface":      "#1b1e26",   # cards
    "surface_alt":  "#20242e",   # inner elements (entries, terminal)
    "border":       "#2b2f3a",
    "accent":       "#4c8dff",
    "accent_hover": "#3a76e0",
    "accent_dim":   "#31415e",   # inactive-but-ready accent
    "success":      "#2ecc71",
    "success_dim":  "#1e3a2a",
    "warning":      "#e67e22",
    "warning_hover":"#d35400",
    "danger":       "#e74c3c",
    "danger_dim":   "#3a2020",
    "muted":        "#555a66",
    "muted_hover":  "#666b78",
    "text_main":    "#e8eaed",
    "text_muted":   "#8b929e",
    "term_bg":      "#0a0b0d",
    "term_text":    "#7ee787",
}

# Consistent spacing scale
S_XS, S_SM, S_MD, S_LG, S_XL = 6, 10, 16, 20, 28
SIDEBAR_WIDTH = 310

TAB_LABELS = [
    "📊  Module 1 · Baseband",
    "📡  Module 2 · Modulation",
    "🌊  Module 3-4 · Spectrum",
    "📶  Module 5 · Channel",
    "🔧  Module 6 · DF2T Filter",
    "🎛️  Module 7 · IIR Design",
    "🔍  Module 8 · Demodulation",
    "🎧  Audio Sync",
]


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
        self.geometry("1340x870")
        self.configure(fg_color=COLORS["bg"])
        self.minsize(1100, 700)

        # Fonts
        self.font_title = ctk.CTkFont(family="Segoe UI", size=24, weight="bold")
        self.font_subtitle = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        self.font_normal = ctk.CTkFont(family="Segoe UI", size=13)
        self.font_small = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_terminal = ctk.CTkFont(family="Consolas", size=13)

        self.pipeline = main.TransceiverPipeline()

        # Grid Layout -- column 0 (sidebar) is locked to a hard minsize so
        # no amount of content stacked inside it can ever push it wider;
        # only column 1 (main content) is allowed to grow.
        self.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.paned_window = tk.PanedWindow(self, orient=tk.VERTICAL, bg=COLORS["bg"], 
                                           bd=0, sashwidth=10, sashcursor="sb_v_double_arrow")
        self.paned_window.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=S_LG, pady=S_LG)

        self.create_sidebar()
        self.create_main_view()
        self.create_terminal_view()

        self.image_references = {}
        self.update_steps(active=None, completed=set())

    # Sidebar

    def create_sidebar(self):
        # A single scrollable frame gridded directly into the locked
        # column. No wrapper frame -- grid_columnconfigure(..., minsize=)
        # on the root window is what actually enforces the fixed width,
        # so nested propagate() tricks aren't needed and can't fight it.
        self.sidebar = ctk.CTkScrollableFrame(
            self, width=SIDEBAR_WIDTH, corner_radius=0, fg_color=COLORS["sidebar"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["muted"],
        )
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        # Force every child to respect the column width even if a widget
        # (e.g. a long decoded-text label) asks for more room.
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Brand header
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.grid(row=0, column=0, pady=(S_XL, S_MD), sticky="ew")
        ctk.CTkLabel(title_frame, text="◈ DSP Modem", font=self.font_title,
                     text_color=COLORS["accent"]).pack()
        ctk.CTkLabel(title_frame, text="Software Defined Acoustic Transceiver",
                     font=self.font_small, text_color=COLORS["text_muted"]).pack(pady=(2, 0))

        #Section 1: Payload
        input_frame = self._card(self.sidebar)
        input_frame.grid(row=1, column=0, padx=S_LG, pady=(0, S_MD), sticky="ew")
        self._card_title(input_frame, "Data Payload")

        self.input_entry = ctk.CTkEntry(
            input_frame, placeholder_text="Enter message...", font=self.font_normal,
            height=36, corner_radius=8, fg_color=COLORS["surface_alt"],
            border_color=COLORS["border"], border_width=1,
        )
        self.input_entry.insert(0, "KU")
        self.input_entry.pack(padx=S_MD, pady=(0, S_MD), fill="x")

        # Section 2: Pipeline control
        control_frame = self._card(self.sidebar)
        control_frame.grid(row=2, column=0, padx=S_LG, pady=0, sticky="ew")
        self._card_title(control_frame, "Pipeline Controls")

        self.btn_step1 = self._button(control_frame, "TX Encode & Modulate",
                                       fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                       command=self.run_step1)
        self.btn_step1.pack(padx=S_MD, pady=(0, S_XS), fill="x")

        self.btn_step2a = self._button(control_frame, "Channel (Simulated)",
                                        fg_color=COLORS["muted"], hover_color=COLORS["muted_hover"],
                                        command=self.run_step2a)
        self.btn_step2a.pack(padx=S_MD, pady=S_XS, fill="x")

        self.btn_step2b = self._button(control_frame, "Acoustic - Play + Record",
                                        fg_color=COLORS["warning_hover"], hover_color=COLORS["warning"],
                                        command=self.run_step2b)
        self.btn_step2b.pack(padx=S_MD, pady=S_XS, fill="x")

        self.btn_step2c = self._button(control_frame, "Acoustic - Record Only",
                                        fg_color="#7d5a00", hover_color="#a07800",
                                        command=self.run_record_only)
        self.btn_step2c.pack(padx=S_MD, pady=S_XS, fill="x")

        self.btn_step3 = self._button(control_frame, "RX Filter & Decode",
                                       fg_color=COLORS["muted"], hover_color=COLORS["muted_hover"],
                                       command=self.run_step3)
        self.btn_step3.pack(padx=S_MD, pady=(S_XS, S_XS), fill="x")

        self.btn_clear = self._button(control_frame, "↺  Reset Pipeline",
                                       fg_color=COLORS["danger"], hover_color="#c0392b",
                                       command=self.clear_results)
        self.btn_clear.pack(padx=S_MD, pady=(S_XS, S_MD), fill="x")

        # Step indicator
        step_frame = self._card(self.sidebar)
        step_frame.grid(row=3, column=0, padx=S_LG, pady=(0, S_MD), sticky="ew")
        self._card_title(step_frame, "Progress")

        indicator_row = ctk.CTkFrame(step_frame, fg_color="transparent")
        indicator_row.pack(padx=S_MD, pady=(0, S_SM), fill="x")
        self.step_dot_labels = {}
        self.step_names = ["Encode", "Channel", "Decode"]
        for i, name in enumerate(self.step_names):
            lbl = ctk.CTkLabel(indicator_row, text=f"○  {name}", font=self.font_small,
                                text_color=COLORS["text_muted"], anchor="w")
            lbl.pack(side="left", expand=True, fill="x")
            self.step_dot_labels[i] = lbl

        self.progress = ctk.CTkProgressBar(step_frame, mode="indeterminate", height=4,
                                            fg_color=COLORS["surface_alt"], progress_color=COLORS["accent"])
        self.progress.pack(padx=S_MD, pady=(0, S_MD), fill="x")
        self.progress.set(0)

        # Section 3: Audio playback
        self.audio_frame = self._card(self.sidebar)
        self.audio_frame.grid(row=4, column=0, padx=S_LG, pady=(0, S_MD), sticky="ew")
        self._card_title(self.audio_frame, "Signal Inspection")

        self.btn_play_tx = self._button(self.audio_frame, "🔊  Play TX Signal", height=32,
                                         fg_color=COLORS["success"], hover_color="#27ae60",
                                         state="disabled", command=self.play_tx)
        self.btn_play_tx.pack(padx=S_MD, pady=(0, S_XS), fill="x")

        self.btn_save_tx = self._button(self.audio_frame, "💾  Save TX as WAV", height=32,
                                         fg_color="#1a6b4a", hover_color="#1f8a5e",
                                         state="disabled", command=self.save_tx_wav)
        self.btn_save_tx.pack(padx=S_MD, pady=S_XS, fill="x")

        self.btn_play_rx = self._button(self.audio_frame, "🔊  Play RX Signal", height=32,
                                         fg_color="#8e44ad", hover_color="#9b59b6",
                                         state="disabled", command=self.play_rx)
        self.btn_play_rx.pack(padx=S_MD, pady=(S_XS, S_MD), fill="x")

        # Section 4: Standalone RX
        self.rx_only_frame = self._card(self.sidebar)
        self.rx_only_frame.grid(row=5, column=0, padx=S_LG, pady=(0, S_MD), sticky="ew")
        self._card_title(self.rx_only_frame, "Standalone Receiver")
        self.rx_duration_entry = ctk.CTkEntry(
            self.rx_only_frame, placeholder_text="Rec. duration (s)", font=self.font_normal,
            height=32, corner_radius=8, fg_color=COLORS["surface_alt"],
            border_color=COLORS["border"], border_width=1,
        )
        self.rx_duration_entry.insert(0, "8.0")
        self.rx_duration_entry.pack(padx=S_MD, pady=(0, S_XS), fill="x")

        self.btn_rx_only = self._button(self.rx_only_frame, "🎙️  Record & Decode (Auto)", height=32,
                                         fg_color="#005b96", hover_color="#004a7a",
                                         command=self.run_receive_only)
        self.btn_rx_only.pack(padx=S_MD, pady=(0, S_MD), fill="x")

        #  Output result
        self.result_frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS["success_dim"],
                                          border_width=1, border_color=COLORS["success"],
                                          corner_radius=10)
        self.result_frame.grid(row=6, column=0, padx=S_LG, pady=(0, S_MD), sticky="ew")
        self.output_label = ctk.CTkLabel(self.result_frame, text="Decoded Output\n— waiting —",
                                          font=self.font_subtitle, text_color="#a8e6cf",
                                          wraplength=SIDEBAR_WIDTH - (2 * S_LG) - S_MD)
        self.output_label.pack(pady=S_MD, padx=S_MD)

        # Status bar
        self.status_var = ctk.StringVar(value="System Ready")
        self.status_label = ctk.CTkLabel(self.sidebar, textvariable=self.status_var,
                                          font=ctk.CTkFont(size=11, weight="bold"),
                                          text_color=COLORS["text_muted"],
                                          wraplength=SIDEBAR_WIDTH - (2 * S_LG))
        self.status_label.grid(row=10, column=0, padx=S_LG, pady=(0, S_LG), sticky="ew")


    # Small style helpers

    def _card(self, parent):
        return ctk.CTkFrame(parent, fg_color=COLORS["surface"], corner_radius=12,
                             border_width=1, border_color=COLORS["border"])

    def _card_title(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=COLORS["text_muted"]).pack(pady=(S_MD, S_SM), padx=S_MD, anchor="w")

    def _button(self, parent, text, height=36, **kwargs):
        return ctk.CTkButton(parent, text=text, font=self.font_normal, height=height,
                              corner_radius=8, **kwargs)

    def update_steps(self, active=None, completed=None):
        completed = completed or set()
        for i, name in enumerate(self.step_names):
            lbl = self.step_dot_labels[i]
            if i in completed:
                lbl.configure(text=f"●  {name}", text_color=COLORS["success"])
            elif i == active:
                lbl.configure(text=f"◐  {name}", text_color=COLORS["accent"])
            else:
                lbl.configure(text=f"○  {name}", text_color=COLORS["text_muted"])


    # Main dashboard

    def create_main_view(self):
        self.main_view = ctk.CTkFrame(self.paned_window, corner_radius=15, fg_color=COLORS["surface"],
                                       border_width=1, border_color=COLORS["border"])
        self.paned_window.add(self.main_view, stretch="always", minsize=200)
        self.main_view.grid_rowconfigure(1, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.main_view, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=S_LG, pady=S_MD)
        ctk.CTkLabel(header, text="Visual Analysis Dashboard", font=self.font_title,
                     text_color=COLORS["text_main"]).pack(side="left")

        self.tabview = ctk.CTkTabview(
            self.main_view, corner_radius=10,
            fg_color=COLORS["surface_alt"],
            segmented_button_fg_color=COLORS["surface_alt"],
            segmented_button_unselected_color=COLORS["surface_alt"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
            text_color=COLORS["text_main"],
        )
        self.tabview.grid(row=1, column=0, padx=S_LG, pady=(0, S_LG), sticky="nsew")

        self.tabs = TAB_LABELS
        self.image_frames = {}
        for tab_name in self.tabs:
            self.tabview.add(tab_name)

            scroll_frame = ctk.CTkScrollableFrame(self.tabview.tab(tab_name), fg_color="transparent")
            scroll_frame.pack(expand=True, fill="both")
            self.image_frames[tab_name] = scroll_frame

            self._empty_state(scroll_frame)

    def _empty_state(self, parent):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(expand=True, fill="both", pady=60)
        ctk.CTkLabel(wrap, text="◌", font=ctk.CTkFont(size=32),
                     text_color=COLORS["text_muted"]).pack()
        ctk.CTkLabel(wrap, text="Run the pipeline to generate visual data.",
                     font=self.font_normal, text_color=COLORS["text_muted"]).pack(pady=(S_SM, 0))


    # Terminal

    def create_terminal_view(self):
        self.term_view = ctk.CTkFrame(self.paned_window, corner_radius=15, fg_color=COLORS["surface"],
                                       border_width=1, border_color=COLORS["border"])
        self.paned_window.add(self.term_view, stretch="always", minsize=100)
        self.term_view.grid_rowconfigure(1, weight=1)
        self.term_view.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.term_view, fg_color="transparent")
        header.grid(row=0, column=0, padx=S_LG, pady=(S_MD, S_XS), sticky="ew")
        ctk.CTkLabel(header, text="●  DSP Process Log", font=self.font_subtitle,
                     text_color=COLORS["text_main"]).pack(side="left")

        self.console_textbox = ctk.CTkTextbox(
            self.term_view, font=self.font_terminal, fg_color=COLORS["term_bg"],
            text_color=COLORS["term_text"], corner_radius=8,
            border_width=1, border_color=COLORS["border"], wrap="word",
        )
        self.console_textbox.grid(row=1, column=0, padx=S_MD, pady=(0, S_MD), sticky="nsew")
        self.console_textbox.configure(state="disabled")

        sys.stdout = StdoutRedirector(self.console_textbox)

    def set_status(self, text, is_working=False):
        self.status_var.set(text)
        if is_working:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)
        self.update()

    # Reset

    def clear_results(self):
        self.btn_step2a.configure(fg_color=COLORS["muted"])
        self.btn_step2b.configure(fg_color=COLORS["warning_hover"])
        self.btn_step2c.configure(fg_color="#7d5a00")
        self.btn_step3.configure(fg_color=COLORS["muted"])
        self.btn_play_tx.configure(state="disabled")
        self.btn_save_tx.configure(state="disabled")
        self.btn_play_rx.configure(state="disabled")

        self.result_frame.configure(border_color=COLORS["success"], fg_color=COLORS["success_dim"])
        self.output_label.configure(text="Decoded Output\n— waiting —", text_color="#a8e6cf")

        self.console_textbox.configure(state="normal")
        self.console_textbox.delete("0.0", "end")
        self.console_textbox.configure(state="disabled")

        for tab_name in self.tabs:
            for widget in self.image_frames[tab_name].winfo_children():
                widget.destroy()
            self._empty_state(self.image_frames[tab_name])
        self.image_references = {}

        self.pipeline = main.TransceiverPipeline()
        self.update_steps(active=None, completed=set())
        self.set_status("System Reset")


    # Pipeline steps

    def run_step1(self):
        txt = self.input_entry.get().strip()
        if not txt:
            return

        def task():
            self.after(0, lambda: self.set_status("Encoding & Modulating...", True))
            self.after(0, lambda: self.update_steps(active=0, completed=set()))
            self.pipeline.encode_and_modulate(txt)

            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[0], [
                os.path.join(config.OUTPUT_DIR, "module_1", "bitstream_stem.png"),
                os.path.join(config.OUTPUT_DIR, "module_1", "convolution_demo.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[1], [
                os.path.join(config.OUTPUT_DIR, "module_2", "bpsk_waveform_zoomed.png"),
                os.path.join(config.OUTPUT_DIR, "module_2", "bpsk_waveform_full.png"),
                os.path.join(config.OUTPUT_DIR, "module_2", "nyquist_compliance.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[2], [
                os.path.join(config.OUTPUT_DIR, "module_3_4", "fft_magnitude_spectrum.png")
            ]))

            self.after(0, lambda: self.btn_step2a.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"]))
            self.after(0, lambda: self.btn_play_tx.configure(state="normal"))
            self.after(0, lambda: self.btn_save_tx.configure(state="normal"))
            self.after(0, lambda: self.update_steps(active=None, completed={0}))
            self.after(0, lambda: self.set_status("Step 1 Complete. Awaiting Channel.", False))

        threading.Thread(target=task, daemon=True).start()

    def run_step2a(self):
        if self.pipeline.tx_signal is None:
            return

        def task():
            self.after(0, lambda: self.set_status("Simulating Multipath & AWGN...", True))
            self.after(0, lambda: self.update_steps(active=1, completed={0}))
            self.pipeline.run_channel_simulation()

            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[3], [
                os.path.join(config.OUTPUT_DIR, "module_5", "channel_effect.png"),
                os.path.join(config.OUTPUT_DIR, "module_5", "pole_zero_map.png"),
                os.path.join(config.OUTPUT_DIR, "module_5", "channel_impulse_response.png")
            ]))
            self.after(0, lambda: self.tabview.set(TAB_LABELS[3]))

            self.after(0, lambda: self.btn_step3.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"]))
            self.after(0, lambda: self.btn_play_rx.configure(state="normal"))
            self.after(0, lambda: self.update_steps(active=None, completed={0, 1}))
            self.after(0, lambda: self.set_status("Simulation Complete. Awaiting Decode.", False))

        threading.Thread(target=task, daemon=True).start()

    def run_step2b(self):
        if self.pipeline.tx_signal is None:
            return

        def task():
            self.after(0, lambda: self.set_status("Acoustic TX/RX Active...", True))
            self.after(0, lambda: self.update_steps(active=1, completed={0}))
            try:
                self.pipeline.run_channel_acoustic()

                self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[7], [
                    os.path.join(config.OUTPUT_DIR, "module_sync", "sync_correlation.png")
                ]))
                self.after(0, lambda: self.tabview.set(TAB_LABELS[7]))

                self.after(0, lambda: self.btn_step3.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"]))
                self.after(0, lambda: self.btn_play_rx.configure(state="normal"))
                self.after(0, lambda: self.update_steps(active=None, completed={0, 1}))
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
            self.after(0, lambda: self.update_steps(active=2, completed={0, 1}))
            recovered = self.pipeline.decode_and_demodulate()

            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[4], [
                os.path.join(config.OUTPUT_DIR, "module_6", "df2t_impulse_response.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[5], [
                os.path.join(config.OUTPUT_DIR, "module_7", "iir_frequency_response.png")
            ]))
            self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[6], [
                os.path.join(config.OUTPUT_DIR, "module_8", "demodulation_waveforms.png")
            ]))
            self.after(0, lambda: self.tabview.set(TAB_LABELS[6]))

            if recovered == self.pipeline.input_text:
                self.after(0, lambda: self.result_frame.configure(border_color=COLORS["success"], fg_color=COLORS["success_dim"]))
                self.after(0, lambda: self.output_label.configure(text=f"✓ Decoded Output\n{recovered}", text_color=COLORS["success"]))
            else:
                self.after(0, lambda: self.result_frame.configure(border_color=COLORS["danger"], fg_color=COLORS["danger_dim"]))
                self.after(0, lambda: self.output_label.configure(text=f"✗ Decoded Output\n{recovered}", text_color=COLORS["danger"]))

            self.after(0, lambda: self.update_steps(active=None, completed={0, 1, 2}))
            self.after(0, lambda: self.set_status("Pipeline Complete!", False))

        threading.Thread(target=task, daemon=True).start()

    def load_images_into_tab(self, tab_name, filepaths):
        for widget in self.image_frames[tab_name].winfo_children():
            widget.destroy()

        for filepath in filepaths:
            if os.path.exists(filepath):
                img = Image.open(filepath)
                w, h = img.size
                ratio = min(750 / w, 500 / h)
                new_w, new_h = int(w * ratio), int(h * ratio)

                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))

                if tab_name not in self.image_references:
                    self.image_references[tab_name] = []
                self.image_references[tab_name].append(ctk_img)

                card = ctk.CTkFrame(self.image_frames[tab_name], fg_color=COLORS["surface"],
                                     corner_radius=10, border_width=1, border_color=COLORS["border"])
                card.pack(pady=S_SM, padx=S_SM, fill="x")
                lbl = ctk.CTkLabel(card, image=ctk_img, text="")
                lbl.pack(pady=S_SM, padx=S_SM)

    # ------------------------------------------------------------------
    # Standalone Receive
    # ------------------------------------------------------------------
    def run_receive_only(self):
        try:
            dur_str = self.rx_duration_entry.get().strip()
            duration = float(dur_str) if dur_str else 8.0
        except ValueError:
            duration = 8.0

        def task():
            self.after(0, lambda: self.set_status("Listening... (play audio now)", True))
            self.after(0, lambda: self.update_steps(active=2, completed=set()))
            try:
                # Record for the user-specified duration
                recovered = self.pipeline.receive_only(duration_s=duration, num_chars=0)

                self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[7], [
                    os.path.join(config.OUTPUT_DIR, "module_sync", "sync_correlation.png")
                ]))
                self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[4], [
                    os.path.join(config.OUTPUT_DIR, "module_6", "df2t_impulse_response.png")
                ]))
                self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[5], [
                    os.path.join(config.OUTPUT_DIR, "module_7", "iir_frequency_response.png")
                ]))
                self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[6], [
                    os.path.join(config.OUTPUT_DIR, "module_8", "demodulation_waveforms.png")
                ]))

                self.after(0, lambda: self.tabview.set(TAB_LABELS[6]))
                self.after(0, lambda: self.btn_play_rx.configure(state="normal"))

                if recovered:
                    self.after(0, lambda: self.result_frame.configure(border_color=COLORS["success"], fg_color=COLORS["success_dim"]))
                    self.after(0, lambda: self.output_label.configure(text=f"✓ Decoded Output\n{recovered}", text_color=COLORS["success"]))
                else:
                    self.after(0, lambda: self.result_frame.configure(border_color=COLORS["danger"], fg_color=COLORS["danger_dim"]))
                    self.after(0, lambda: self.output_label.configure(text=f"✗ Decoded Output\n{recovered}", text_color=COLORS["danger"]))

                self.after(0, lambda: self.update_steps(active=None, completed={2}))
                self.after(0, lambda: self.set_status("Standalone RX Complete!", False))

            except Exception as e:
                print(f"RX Error: {e}")
                self.after(0, lambda: self.set_status(f"Error: {str(e)}", False))

        threading.Thread(target=task, daemon=True).start()

    def play_tx(self):
        if self.pipeline.tx_signal is not None:
            sig = self.pipeline.tx_signal * 0.8 / np.max(np.abs(self.pipeline.tx_signal))
            sd.play(sig, self.pipeline.fs)

    def play_rx(self):
        if self.pipeline.rx_signal is not None:
            sig = self.pipeline.rx_signal * 0.8 / np.max(np.abs(self.pipeline.rx_signal))
            sd.play(sig, self.pipeline.fs)

    # Save TX audio as WAV (full transmission: silence + preamble + payload + silence)
    def save_tx_wav(self):
        if self.pipeline.tx_signal is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save TX Signal (for external playback)",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            initialfile=f"tx_{self.pipeline.input_text or 'signal'}.wav",
        )
        if not path:
            return

        # Build the same full_tx layout that transmit_and_receive plays,
        # so the receiver's cross-correlator can find the preamble.
        from modules import module_audio_io
        preamble = module_audio_io.modulate_sync_preamble(
            self.pipeline.fs, self.pipeline.fc, config.SAMPLES_PER_BIT
        )
        silence = np.zeros(int(self.pipeline.fs * 0.5))
        full_tx = np.concatenate([silence, preamble, self.pipeline.tx_signal, silence])
        full_tx = full_tx * 0.8 / np.max(np.abs(full_tx))

        sig_int16 = (full_tx * 32767).astype(np.int16)
        wavfile.write(path, self.pipeline.fs, sig_int16)
        duration_s = len(full_tx) / self.pipeline.fs
        print(f"[Save] Full TX saved ({duration_s:.2f}s, includes preamble) -> {path}")
        self.set_status(f"Saved: {os.path.basename(path)}", False)

    # Record Only (play nothing; capture from mic for external playback)

    def run_record_only(self):
        if self.pipeline.tx_signal is None:
            return

        def task():
            self.after(0, lambda: self.set_status("Recording... (waiting for signal)", True))
            self.after(0, lambda: self.update_steps(active=1, completed={0}))
            try:
                rx = self.pipeline.run_record_only()

                self.after(0, lambda: self.load_images_into_tab(TAB_LABELS[7], [
                    os.path.join(config.OUTPUT_DIR, "module_sync", "sync_correlation.png")
                ]))
                self.after(0, lambda: self.tabview.set(TAB_LABELS[7]))
                self.after(0, lambda: self.btn_step3.configure(fg_color=COLORS["accent"],
                                                                hover_color=COLORS["accent_hover"]))
                self.after(0, lambda: self.btn_play_rx.configure(state="normal"))
                self.after(0, lambda: self.update_steps(active=None, completed={0, 1}))
                self.after(0, lambda: self.set_status("Recording Complete. Awaiting Decode.", False))
            except Exception as e:
                print(f"Record Error: {e}")
                self.after(0, lambda: self.set_status(f"Error: {str(e)}", False))

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    app = ModernTransceiverUI()
    app.mainloop()