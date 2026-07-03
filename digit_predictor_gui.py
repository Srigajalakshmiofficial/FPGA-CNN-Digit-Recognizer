"""
digit_predictor_gui.py
======================
Draw-and-Predict MNIST digit GUI using the Quartus INT4 hex files.
Mirrors the Verilog hardware (mnist_top.v) exactly:
  - Architecture : 784 → 128 → 64 → 32 → 10
  - Weights      : INT4 packed (2 per byte), low nibble = even index
  - Biases       : INT16, one 4-hex-char value per line
  - Activation   : ReLU clamped to [0, 127]  (signed 8-bit)
  - Pixels       : scaled to [0, 127]  (Verilog: reg signed [7:0])

Hex files loaded from: ./quartus_hex_28_99/

Run:
    python digit_predictor_gui.py

Requirements:
    pip install pillow numpy
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEX_DIR    = os.path.join(SCRIPT_DIR, "quartus_hex_28_99")

def hex_path(name):
    return os.path.join(HEX_DIR, name)

# ─── Weight / Bias Loaders ────────────────────────────────────────────────────

def load_int4_weights(filename, n_rows, n_cols):
    """
    Load packed INT4 weight matrix from Quartus hex file.
    Each byte holds two INT4 values:
      lo nibble (bits[3:0]) → even flat index
      hi nibble (bits[7:4]) → odd  flat index
    flat index = neuron * n_cols + input_idx
    Returns numpy float32 array (n_rows, n_cols).
    """
    path = hex_path(filename)
    raw = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("//") and not s.startswith("@"):
                raw.append(int(s, 16))

    weights = []
    for byte in raw:
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        # Sign-extend 4-bit
        lo = lo if lo < 8 else lo - 16
        hi = hi if hi < 8 else hi - 16
        weights.append(lo)
        weights.append(hi)

    n_weights = n_rows * n_cols
    weights = weights[:n_weights]
    if len(weights) < n_weights:
        weights += [0] * (n_weights - len(weights))
    return np.array(weights, dtype=np.float32).reshape(n_rows, n_cols)


def load_int16_biases(filename, count):
    """
    Load INT16 biases from Quartus hex file (one 4-hex-char entry per line).
    Returns numpy float32 array (count,).
    """
    path = hex_path(filename)
    biases = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("//") and not s.startswith("@"):
                v = int(s, 16)
                if v >= 0x8000:
                    v -= 0x10000
                biases.append(v)
    biases = biases[:count]
    if len(biases) < count:
        biases += [0] * (count - len(biases))
    return np.array(biases, dtype=np.float32)


# ─── MLP Inference (exact Verilog mirror) ────────────────────────────────────

class MLPInfer:
    """
    Pure-Python INT4 MLP that replicates the Verilog mlp_infer module.
    Architecture: 784 → 128 → 64 → 32 → 10
    """

    def __init__(self):
        self._loaded = False
        self.status  = "Loading weights…"

    def load(self):
        try:
            self.W1 = load_int4_weights("fc1_weights.hex", 128, 784)
            self.b1 = load_int16_biases("fc1_bias.hex",    128)
            self.W2 = load_int4_weights("fc2_weights.hex",  64, 128)
            self.b2 = load_int16_biases("fc2_bias.hex",     64)
            self.W3 = load_int4_weights("fc3_weights.hex",  32,  64)
            self.b3 = load_int16_biases("fc3_bias.hex",     32)
            self.W4 = load_int4_weights("fc4_weights.hex",  10,  32)
            self.b4 = load_int16_biases("fc4_bias.hex",     10)
            self._loaded = True
            self.status  = "✓  Weights loaded from quartus_hex_28_99"
        except Exception as e:
            self.status = f"Load error: {e}"

    # ReLU clamped to [0, 127]  (Verilog: if acc<=0 → 0, if acc>127 → 127)
    @staticmethod
    def _relu8(x):
        return np.clip(x, 0, 127).astype(np.int32)

    def infer(self, pixels_uint8):
        """
        pixels_uint8 : numpy array (784,) uint8, values in [0, 127].
        Mirrors Verilog mlp_infer exactly:
          - x_val is `reg signed [7:0]`  → pixels must be 0-127
          - acc is `reg signed [22:0]`   → 23-bit accumulator (int32 OK)
          - logits stored as acc[15:0]   → 16-bit signed truncation!
          - ReLU clamps activations to [0, 127]
        Returns (predicted_digit : int, logits : ndarray shape (10,)).
        """
        if not self._loaded:
            return None, None

        # Cast exactly as Verilog: reg signed [7:0]
        x = pixels_uint8.astype(np.int8).astype(np.int32)

        # Integer MACs to match hardware precisely
        a1 = self._relu8(self.W1.astype(np.int32) @ x  + self.b1.astype(np.int32))
        a2 = self._relu8(self.W2.astype(np.int32) @ a1 + self.b2.astype(np.int32))
        a3 = self._relu8(self.W3.astype(np.int32) @ a2 + self.b3.astype(np.int32))

        # FC4: Verilog stores logit[n] = acc[15:0]  ← 16-bit signed truncation!
        logits_raw   = self.W4.astype(np.int32) @ a3 + self.b4.astype(np.int32)
        logits_trunc = logits_raw & 0xFFFF
        logits_trunc = np.where(logits_trunc >= 0x8000,
                                logits_trunc - 0x10000,
                                logits_trunc).astype(np.int32)

        digit = int(np.argmax(logits_trunc))
        return digit, logits_trunc.astype(np.float32)


# ─── Preprocessing ────────────────────────────────────────────────────────────
#
# Matches what the PC-side client does before sending 784 bytes over UART.
# Key constraint: scale to [0, 127]  (not 0-255) because hardware x_val
# is a signed 8-bit register.

def preprocess(pil_img):
    """
    MNIST-style pipeline:
      1. Slight Gaussian blur  (denoise)
      2. Bounding-box crop + 15 % padding
      3. Fit inside 20×20  (preserve aspect ratio)
      4. Center-paste onto 28×28 canvas
      5. Final light blur
      6. Scale to [0, 127]
    Returns uint8 array (784,).
    """
    img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    arr = np.array(img, dtype=np.uint8)

    thresh = 20
    mask   = arr > thresh
    rows   = np.any(mask, axis=1)
    cols   = np.any(mask, axis=0)

    if not rows.any():
        return np.zeros(784, dtype=np.uint8)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h      = max(rmax - rmin + 1, 1)
    w      = max(cmax - cmin + 1, 1)
    pad_r  = max(int(h * 0.15), 2)
    pad_c  = max(int(w * 0.15), 2)
    r0     = max(rmin - pad_r, 0)
    r1     = min(rmax + pad_r + 1, arr.shape[0])
    c0     = max(cmin - pad_c, 0)
    c1     = min(cmax + pad_c + 1, arr.shape[1])
    cropped = Image.fromarray(arr[r0:r1, c0:c1])

    TARGET = 20
    cw, ch = cropped.size
    scale  = TARGET / max(cw, ch)
    new_w  = max(int(cw * scale), 1)
    new_h  = max(int(ch * scale), 1)
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas  = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.5))

    out = np.array(canvas, dtype=np.float32) / 255.0
    out = np.clip(out * 127, 0, 127).astype(np.uint8)
    return out.flatten()


# ─── GUI ──────────────────────────────────────────────────────────────────────

class DigitPredictorGUI:
    # ── Colour palette ────────────────────────────────────────────────────────
    P = {
        "bg":       "#0b0b14",
        "panel":    "#13131e",
        "card":     "#1a1a28",
        "card2":    "#1f1f30",
        "accent":   "#7c6fcd",
        "accent2":  "#00e5b0",
        "err":      "#ff4466",
        "text":     "#e8e8f2",
        "dim":      "#4a4a62",
        "canvas_bg":"#05050c",
        "yellow":   "#ffd460",
        "bar_bg":   "#20202e",
        "border":   "#2a2a40",
    }
    # One distinct colour per digit 0-9
    DIGIT_CLR = [
        "#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c",
        "#4dabf7", "#748ffc", "#da77f2", "#f783ac",
        "#63e6be", "#74c0fc",
    ]

    CANVAS_SZ = 312   # drawing canvas size (pixels)
    PEN_W     = 22    # default brush width

    def __init__(self, root: tk.Tk):
        self.root    = root
        self.root.title("Quartus INT4 MLP — Digit Predictor  |  784→128→64→32→10")
        self.root.configure(bg=self.P["bg"])
        self.root.resizable(False, False)

        self.mlp      = MLPInfer()
        self.pen_w    = self.PEN_W
        self.drawing  = False
        self.lx = self.ly = 0
        self.pil_img  = Image.new('L', (self.CANVAS_SZ, self.CANVAS_SZ), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

        self._build_ui()
        # Load weights in background after the window appears
        self.root.after(150, self._kick_load)

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        P = self.P

        # ── Top header bar ───────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=P["panel"], height=54)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(
            hdr,
            text="⬛  Quartus INT4 MLP — Digit Predictor",
            font=("Consolas", 12, "bold"),
            bg=P["panel"], fg=P["accent"]
        ).pack(side=tk.LEFT, padx=18, pady=10)

        self.lbl_status = tk.Label(
            hdr,
            text="Loading hex files…",
            font=("Consolas", 8),
            bg=P["panel"], fg=P["dim"]
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=18)

        # thin accent line below header
        tk.Frame(self.root, bg=P["accent"], height=2).pack(fill=tk.X)

        # ── Main body ────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=P["bg"])
        body.pack(fill=tk.BOTH, padx=18, pady=14)

        self._build_left_panel(body)
        self._build_right_panel(body)

        # ── Status bar ───────────────────────────────────────────────────────
        tk.Frame(self.root, bg=P["border"], height=1).pack(fill=tk.X)
        self.log_var = tk.StringVar(value="→  Draw a digit on the canvas, then press PREDICT")
        tk.Label(
            self.root,
            textvariable=self.log_var,
            font=("Consolas", 8),
            bg=P["panel"], fg=P["dim"],
            anchor="w", padx=18, pady=6
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_left_panel(self, parent):
        P = self.P
        left = tk.Frame(parent, bg=P["bg"])
        left.pack(side=tk.LEFT, padx=(0, 18))

        # Section label
        tk.Label(
            left, text="DRAW DIGIT  (0 – 9)",
            font=("Consolas", 8, "bold"),
            bg=P["bg"], fg=P["dim"]
        ).pack(anchor="w", pady=(0, 6))

        # Glowing border frame around canvas
        outer = tk.Frame(left, bg=P["accent"], padx=2, pady=2)
        outer.pack()
        inner = tk.Frame(outer, bg=P["card"], padx=4, pady=4)
        inner.pack()

        self.canvas = tk.Canvas(
            inner,
            width=self.CANVAS_SZ, height=self.CANVAS_SZ,
            bg=P["canvas_bg"], cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Brush size slider
        ctrl = tk.Frame(left, bg=P["bg"])
        ctrl.pack(fill=tk.X, pady=(10, 0))
        tk.Label(ctrl, text="BRUSH", font=("Consolas", 7), bg=P["bg"], fg=P["dim"]).pack(side=tk.LEFT)
        self.brush_slider = tk.Scale(
            ctrl,
            from_=8, to=36, orient=tk.HORIZONTAL,
            command=self._on_brush_change,
            bg=P["bg"], fg=P["dim"],
            troughcolor=P["card"], highlightthickness=0,
            bd=0, sliderrelief=tk.FLAT, sliderlength=16,
            length=160
        )
        self.brush_slider.set(self.pen_w)
        self.brush_slider.pack(side=tk.LEFT, padx=(6, 0))

        # 28×28 preview
        tk.Label(
            left, text="28 × 28  PREVIEW  (what the model sees)",
            font=("Consolas", 7), bg=P["bg"], fg=P["dim"]
        ).pack(anchor="w", pady=(12, 3))

        preview_outer = tk.Frame(left, bg=P["border"], padx=1, pady=1)
        preview_outer.pack(anchor="w")
        self.preview_canvas = tk.Canvas(
            preview_outer,
            width=140, height=140,
            bg=P["canvas_bg"], highlightthickness=0
        )
        self.preview_canvas.pack()

        # Buttons
        btns = tk.Frame(left, bg=P["bg"])
        btns.pack(pady=12)

        tk.Button(
            btns, text="  CLEAR  ",
            command=self._clear,
            font=("Consolas", 10, "bold"),
            bg=P["card2"], fg=P["dim"],
            activebackground=P["border"], activeforeground=P["text"],
            relief=tk.FLAT, padx=14, pady=8, cursor="hand2", bd=0
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            btns, text="▶  PREDICT",
            command=self._predict,
            font=("Consolas", 10, "bold"),
            bg=P["accent"], fg="#ffffff",
            activebackground="#6a5ec0", activeforeground="#ffffff",
            relief=tk.FLAT, padx=14, pady=8, cursor="hand2", bd=0
        ).pack(side=tk.LEFT)

    def _build_right_panel(self, parent):
        P = self.P
        right = tk.Frame(parent, bg=P["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Big prediction card ───────────────────────────────────────────────
        pred_card = tk.Frame(right, bg=P["card"], padx=22, pady=18)
        pred_card.pack(fill=tk.X)

        tk.Label(
            pred_card, text="PREDICTION",
            font=("Consolas", 8, "bold"),
            bg=P["card"], fg=P["dim"]
        ).pack(anchor="w")

        self.lbl_digit = tk.Label(
            pred_card, text="—",
            font=("Consolas", 90, "bold"),
            bg=P["card"], fg=P["yellow"],
            width=3
        )
        self.lbl_digit.pack()

        self.lbl_conf = tk.Label(
            pred_card, text="",
            font=("Consolas", 9),
            bg=P["card"], fg=P["dim"]
        )
        self.lbl_conf.pack()

        # ── Hardware note ────────────────────────────────────────────────────
        info_card = tk.Frame(right, bg=P["card2"], padx=12, pady=8)
        info_card.pack(fill=tk.X, pady=(6, 0))
        tk.Label(
            info_card,
            text="⚙  INT4 weights  ·  INT16 biases  ·  ReLU[0,127]\n"
                 "   quartus_hex_28_99  ·  784→128→64→32→10",
            font=("Consolas", 7),
            bg=P["card2"], fg=P["dim"],
            justify="left"
        ).pack(anchor="w")

        # ── Logit bar chart ───────────────────────────────────────────────────
        tk.Label(
            right,
            text="LOGIT SCORES   (raw FC4 output · argmax = prediction)",
            font=("Consolas", 7, "bold"),
            bg=P["bg"], fg=P["dim"]
        ).pack(anchor="w", pady=(14, 4))

        self.bar_frame = tk.Frame(right, bg=P["bg"])
        self.bar_frame.pack(fill=tk.X)
        self._build_bars()

    def _build_bars(self):
        P = self.P
        self.bar_rows = []
        for i in range(10):
            row = tk.Frame(self.bar_frame, bg=P["bg"])
            row.pack(fill=tk.X, pady=2)

            lbl_d = tk.Label(
                row, text=str(i), width=2,
                font=("Consolas", 9, "bold"),
                bg=P["bg"], fg=self.DIGIT_CLR[i], anchor="e"
            )
            lbl_d.pack(side=tk.LEFT, padx=(0, 8))

            bar_bg = tk.Frame(row, bg=P["bar_bg"], height=16)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=P["dim"], height=16)
            bar_fill.place(x=0, y=0, width=0, relheight=1)

            lbl_v = tk.Label(
                row, text="", width=10,
                font=("Consolas", 8),
                bg=P["bg"], fg=P["dim"], anchor="w"
            )
            lbl_v.pack(side=tk.LEFT, padx=(8, 0))

            self.bar_rows.append((bar_bg, bar_fill, lbl_v))

    # ── Drawing events ────────────────────────────────────────────────────────

    def _on_press(self, e):
        self.drawing = True
        self.lx, self.ly = e.x, e.y
        r = self.pen_w // 2
        self.canvas.create_oval(
            e.x - r, e.y - r, e.x + r, e.y + r,
            fill="white", outline=""
        )
        self.pil_draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def _on_drag(self, e):
        if not self.drawing:
            return
        self.canvas.create_line(
            self.lx, self.ly, e.x, e.y,
            fill="white", width=self.pen_w,
            capstyle=tk.ROUND, smooth=True
        )
        self.pil_draw.line(
            [(self.lx, self.ly), (e.x, e.y)],
            fill=255, width=self.pen_w
        )
        self.lx, self.ly = e.x, e.y

    def _on_release(self, e):
        self.drawing = False
        self._update_preview()  # live 28×28 preview after each stroke

    def _on_brush_change(self, val):
        self.pen_w = int(val)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _update_preview(self, pixels=None):
        """Render the 28×28 image the model actually processes."""
        if pixels is not None:
            arr28 = pixels.reshape(28, 28).astype(np.float32)
            arr28 = np.clip(arr28 * 2, 0, 255).astype(np.uint8)
        else:
            arr28 = np.array(
                self.pil_img.resize((28, 28), Image.Resampling.LANCZOS),
                dtype=np.uint8
            )

        big = np.repeat(np.repeat(arr28, 5, axis=0), 5, axis=1)  # 140×140
        self._tk_prev = tk.PhotoImage(width=140, height=140)
        rows = []
        for row in big:
            row_str = " ".join(f"#{v:02x}{v:02x}{v:02x}" for v in row)
            rows.append("{" + row_str + "}")
        self._tk_prev.put(" ".join(rows))
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self._tk_prev)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _clear(self):
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self.pil_img  = Image.new('L', (self.CANVAS_SZ, self.CANVAS_SZ), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.lbl_digit.config(text="—", fg=self.P["yellow"])
        self.lbl_conf.config(text="")
        for _, bar_fill, lbl_v in self.bar_rows:
            bar_fill.place(width=0)
            lbl_v.config(text="", fg=self.P["dim"])
        self._log("Cleared — draw a new digit")

    def _predict(self):
        if not self.mlp._loaded:
            self._log("⚠  Weights still loading — please wait a moment"); return
        self._log("Running INT4 inference…")
        threading.Thread(target=self._run_infer, daemon=True).start()

    def _run_infer(self):
        pixels = preprocess(self.pil_img)
        digit, logits = self.mlp.infer(pixels)
        self.root.after(0, self._show_result, digit, logits, pixels)

    def _show_result(self, digit, logits, pixels):
        if digit is None:
            self._log("✗  Inference failed — check hex files"); return

        P = self.P
        clr = self.DIGIT_CLR[digit % 10]
        self.lbl_digit.config(text=str(digit), fg=clr)

        # Softmax for human-readable confidence
        shifted = logits - logits.max()
        exp     = np.exp(np.clip(shifted, -80, 0))
        probs   = exp / exp.sum()
        conf    = probs[digit] * 100

        self.lbl_conf.config(
            text=f"Confidence  {conf:.1f}%     logit  {logits[digit]:+.1f}",
            fg=clr
        )

        # Bar chart
        lo, hi = logits.min(), logits.max()
        span   = max(hi - lo, 1.0)
        for i, (bar_bg, bar_fill, lbl_v) in enumerate(self.bar_rows):
            W    = bar_bg.winfo_width()
            if W < 4: W = 180
            frac = max((logits[i] - lo) / span, 0.0)
            fw   = int(frac * W)
            is_pred = (i == digit)
            bar_fill.config(bg=self.DIGIT_CLR[i] if is_pred else P["dim"])
            bar_fill.place(width=fw)
            lbl_v.config(
                text=f"{logits[i]:+7.1f}  {probs[i]*100:5.1f}%",
                fg=self.DIGIT_CLR[i] if is_pred else P["dim"]
            )

        # Update preview to show exactly what was fed to the model
        self._update_preview(pixels)
        active = int((pixels > 0).sum())
        self._log(
            f"✓  Predicted: {digit}   confidence: {conf:.1f}%   "
            f"active pixels: {active}/784   logit: {logits[digit]:+.1f}"
        )

    # ── Weight Loading ────────────────────────────────────────────────────────

    def _kick_load(self):
        def _bg():
            self.mlp.load()
            self.root.after(0, self._on_loaded)
        threading.Thread(target=_bg, daemon=True).start()

    def _on_loaded(self):
        ok  = self.mlp._loaded
        clr = self.P["accent2"] if ok else self.P["err"]
        self.lbl_status.config(text=self.mlp.status, fg=clr)
        self._log(self.mlp.status)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_var.set(f"→  {msg}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()

    # Try to set a modern DPI-aware icon title
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass

    app = DigitPredictorGUI(root)
    root.mainloop()
