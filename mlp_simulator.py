"""
mlp_simulator.py
=================
Pure-Python INT4 MLP simulator that mirrors the Verilog hardware exactly.
Architecture : 784 → 128 → 64 → 32 → 10  (INT4 weights, INT16 biases)
Hex files    : fc1_weights.hex  (50176 bytes, 2 INT4 per byte)
               fc2_weights.hex  (4096  bytes)
               fc3_weights.hex  (1024  bytes)
               fc4_weights.hex  (160   bytes)
               fc1_bias.hex     (128 × 16-bit)
               fc2_bias.hex     (64  × 16-bit)
               fc3_bias.hex     (32  × 16-bit)
               fc4_bias.hex     (10  × 16-bit)

Run: python mlp_simulator.py
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

def hex_path(name):
    return os.path.join(SCRIPT_DIR, name)

# ─── Weight / Bias Loaders ────────────────────────────────────────────────────

def load_int4_weights(filename, n_rows, n_cols):
    """
    Load packed INT4 weight matrix from hex file.
    Each byte = two INT4 values: low nibble → even index, high nibble → odd index.
    Storage order (matches Verilog flat index): flat = neuron * n_cols + input
    Returns numpy float32 array shape (n_rows, n_cols).
    """
    path = hex_path(filename)
    raw = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                raw.append(int(line, 16))

    n_weights = n_rows * n_cols
    # Unpack pairs: even weight from low nibble, odd from high nibble
    weights = []
    for byte in raw:
        lo = byte & 0x0F          # even index
        hi = (byte >> 4) & 0x0F  # odd index
        # Sign-extend 4-bit to Python int
        lo = lo if lo < 8 else lo - 16
        hi = hi if hi < 8 else hi - 16
        weights.append(lo)
        weights.append(hi)

    weights = weights[:n_weights]
    return np.array(weights, dtype=np.float32).reshape(n_rows, n_cols)


def load_int16_biases(filename, count):
    """
    Load INT16 biases from hex file (one 4-hex-char value per line).
    Returns numpy float32 array shape (count,).
    """
    path = hex_path(filename)
    biases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                v = int(line, 16)
                # Sign-extend 16-bit
                if v >= 0x8000:
                    v -= 0x10000
                biases.append(v)
    return np.array(biases[:count], dtype=np.float32)


# ─── MLP Inference (mirrors Verilog exactly) ─────────────────────────────────

class MLPInfer:
    def __init__(self):
        self._loaded = False
        self.status = "Loading weights..."

    def load(self):
        try:
            self.W1 = load_int4_weights("fc1_weights.hex", 128, 784)
            self.b1 = load_int16_biases("fc1_bias.hex",    128)
            self.W2 = load_int4_weights("fc2_weights.hex",  64, 128)
            self.b2 = load_int16_biases("fc2_bias.hex",     64)
            self.W3 = load_int4_weights("fc3_weights.hex",  32, 64)
            self.b3 = load_int16_biases("fc3_bias.hex",     32)
            self.W4 = load_int4_weights("fc4_weights.hex",  10, 32)
            self.b4 = load_int16_biases("fc4_bias.hex",     10)
            self._loaded = True
            self.status = "Weights loaded ✓"
        except Exception as e:
            self.status = f"Load error: {e}"

    @staticmethod
    def _relu8(x):
        """ReLU clipped to [0, 127] — matches Verilog: if acc<=0→0, if acc>127→127."""
        return np.clip(x, 0, 127).astype(np.int32)

    @staticmethod
    def _sign23(x):
        """Clamp int32 accumulator to 23-bit signed range [-4194304, 4194303]."""
        x = x & 0x7FFFFF | (-(x & 0x400000) & ~0x3FFFFF)
        # Simpler: just keep as int32 — Python int32 arithmetic handles this
        # The key constraint is that logits are stored as acc[15:0] (16-bit signed)
        return x

    def infer(self, pixels_uint8):
        """
        pixels_uint8: numpy array shape (784,) uint8 values 0-127.
        Mirrors Verilog exactly:
          - x_val is signed [7:0]  → pixels must be 0-127
          - acc is signed [22:0]   → 23-bit accumulator
          - logits stored as acc[15:0] → 16-bit signed truncation!
          - ReLU clamps activations to [0, 127]
        Returns (digit, logits_array).
        """
        if not self._loaded:
            return None, None

        # Cast pixels exactly as Verilog: reg signed [7:0]
        # Values 0-127 stay positive; 128-255 would wrap to negative (avoid them)
        x = pixels_uint8.astype(np.int8).astype(np.int32)

        # FC1: acc = bias + sum(w * x),  ReLU → [0,127]  (INT8 activation)
        W1i = self.W1.astype(np.int32)
        b1i = self.b1.astype(np.int32)
        a1_raw = W1i @ x + b1i                    # int32 accumulator
        a1 = self._relu8(a1_raw)                  # clamped to [0,127], int32

        # FC2: same pattern
        W2i = self.W2.astype(np.int32)
        b2i = self.b2.astype(np.int32)
        a2_raw = W2i @ a1 + b2i
        a2 = self._relu8(a2_raw)

        # FC3
        W3i = self.W3.astype(np.int32)
        b3i = self.b3.astype(np.int32)
        a3_raw = W3i @ a2 + b3i
        a3 = self._relu8(a3_raw)

        # FC4 — logits: Verilog stores acc[15:0] (16-bit signed truncation)
        W4i = self.W4.astype(np.int32)
        b4i = self.b4.astype(np.int32)
        logits_raw = W4i @ a3 + b4i               # int32 full accumulator
        # Mirror Verilog: logit[n] = acc[15:0] — take low 16 bits, sign-extend
        logits_trunc = logits_raw.astype(np.int32) & 0xFFFF
        # Sign-extend 16-bit to 32-bit
        logits_trunc = np.where(logits_trunc >= 0x8000,
                                logits_trunc - 0x10000,
                                logits_trunc).astype(np.int32)

        digit = int(np.argmax(logits_trunc))
        return digit, logits_trunc.astype(np.float32)


# ─── Preprocessing ───────────────────────────────────────────────────────────
#
# KEY INSIGHT: Verilog declares x_val as `reg signed [7:0]`.
# This means pixel values MUST be in [0, 127]:
#   - 0   → signed 0   (background/black)
#   - 127 → signed 127 (bright foreground)
#   - 255 → signed -1  (WRONG — negative, ruins inference)
#
# Fix: scale 0-255 grayscale to 0-127 so background=0, foreground≤127.

def preprocess(pil_img):
    """
    MNIST-style preprocessing pipeline:
      1. Slight blur to clean up jagged edges
      2. Find bounding box of drawn pixels
      3. Crop + add 15% padding around the digit
      4. Resize the digit to fit in 20×20 (preserving aspect ratio)
      5. Paste centered into a 28×28 canvas (MNIST standard layout)
      6. Final Gaussian smooth
      7. Scale to [0, 127]  ← required by Verilog signed [7:0] pixel
    Returns numpy uint8 array shape (784,).
    """
    # Step 1: slight denoise
    img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    arr = np.array(img, dtype=np.uint8)

    # Step 2: find bounding box (pixels above threshold)
    thresh = 20
    mask = arr > thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        # Canvas is blank — return all zeros
        return np.zeros(784, dtype=np.uint8)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Step 3: crop with 15% padding
    h = max(rmax - rmin + 1, 1)
    w = max(cmax - cmin + 1, 1)
    pad_r = max(int(h * 0.15), 2)
    pad_c = max(int(w * 0.15), 2)
    r0 = max(rmin - pad_r, 0)
    r1 = min(rmax + pad_r + 1, arr.shape[0])
    c0 = max(cmin - pad_c, 0)
    c1 = min(cmax + pad_c + 1, arr.shape[1])
    cropped = Image.fromarray(arr[r0:r1, c0:c1])

    # Step 4: resize to fit inside 20×20, preserving aspect ratio
    TARGET = 20
    cw, ch = cropped.size
    scale = TARGET / max(cw, ch)
    new_w = max(int(cw * scale), 1)
    new_h = max(int(ch * scale), 1)
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Step 5: paste into 28×28 centered canvas
    canvas = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # Step 6: final smooth (matches MNIST anti-aliasing)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Step 7: scale to [0, 127]
    out = np.array(canvas, dtype=np.float32) / 255.0
    out = np.clip(out * 127, 0, 127).astype(np.uint8)
    return out.flatten()


# ─── GUI ──────────────────────────────────────────────────────────────────────

class SimulatorGUI:
    P = {
        "bg":      "#0d0d12",
        "panel":   "#16161f",
        "card":    "#1c1c28",
        "accent":  "#7c6fcd",
        "accent2": "#00d4aa",
        "err":     "#ff4466",
        "text":    "#e8e8f0",
        "dim":     "#55556a",
        "canvas":  "#050508",
        "yellow":  "#ffd460",
        "bar_bg":  "#252535",
    }
    DIGIT_COLORS = [
        "#ff6b6b","#ffa94d","#ffd43b","#69db7c",
        "#4dabf7","#748ffc","#da77f2","#f783ac",
        "#63e6be","#74c0fc",
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("MLP INT4 Simulator — 784→128→64→32→10")
        self.root.configure(bg=self.P["bg"])
        self.root.resizable(False, False)

        self.mlp = MLPInfer()
        self.CW = self.CH = 308
        self.pen_w = 20
        self.drawing = False
        self.lx = self.ly = 0
        self.pil_img = Image.new('L', (self.CW, self.CH), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.last_logits = None

        self._build_ui()
        self.root.after(100, self._load_weights)

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        P = self.P

        # ── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=P["panel"], height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="▮  MLP INT4 Simulator",
                 font=("Courier New", 13, "bold"),
                 bg=P["panel"], fg=P["accent"]).pack(side=tk.LEFT, padx=16, pady=12)
        self.lbl_status = tk.Label(hdr, text="Loading…",
                                   font=("Courier New", 9),
                                   bg=P["panel"], fg=P["dim"])
        self.lbl_status.pack(side=tk.RIGHT, padx=16)

        # ── Body: left canvas + right panel ─────────────────────────────────
        body = tk.Frame(self.root, bg=P["bg"])
        body.pack(fill=tk.BOTH, padx=16, pady=12)

        # ── Left: drawing area ───────────────────────────────────────────────
        left = tk.Frame(body, bg=P["bg"])
        left.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(left, text="DRAW DIGIT  (0 – 9)",
                 font=("Courier New", 8, "bold"),
                 bg=P["bg"], fg=P["dim"]).pack(anchor="w", pady=(0, 6))

        border = tk.Frame(left, bg=P["accent"], padx=2, pady=2)
        border.pack()
        inner = tk.Frame(border, bg=P["card"], padx=3, pady=3)
        inner.pack()
        self.canvas = tk.Canvas(inner, width=self.CW, height=self.CH,
                                bg=P["canvas"], cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",       self._start)
        self.canvas.bind("<B1-Motion>",      self._move)
        self.canvas.bind("<ButtonRelease-1>",self._end)

        # 28×28 preview strip
        tk.Label(left, text="28×28 PREVIEW",
                 font=("Courier New", 7),
                 bg=P["bg"], fg=P["dim"]).pack(anchor="w", pady=(10, 3))
        self.preview_canvas = tk.Canvas(left, width=140, height=140,
                                        bg=P["canvas"], highlightthickness=1,
                                        highlightbackground=P["dim"])
        self.preview_canvas.pack(anchor="w")

        # Buttons
        btns = tk.Frame(left, bg=P["bg"])
        btns.pack(pady=10)
        tk.Button(btns, text="  CLEAR  ", command=self._clear,
                  font=("Courier New", 10, "bold"),
                  bg=P["card"], fg=P["dim"],
                  relief=tk.FLAT, padx=12, pady=7, cursor="hand2", bd=0
                  ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btns, text="▶  PREDICT", command=self._predict,
                  font=("Courier New", 10, "bold"),
                  bg=P["accent"], fg=P["bg"],
                  relief=tk.FLAT, padx=12, pady=7, cursor="hand2", bd=0
                  ).pack(side=tk.LEFT)

        # ── Right: result panel ──────────────────────────────────────────────
        right = tk.Frame(body, bg=P["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Big digit display
        res_card = tk.Frame(right, bg=P["card"], padx=20, pady=16)
        res_card.pack(fill=tk.X)
        tk.Label(res_card, text="PREDICTION",
                 font=("Courier New", 8, "bold"),
                 bg=P["card"], fg=P["dim"]).pack(anchor="w")
        self.lbl_digit = tk.Label(res_card, text="—",
                                  font=("Courier New", 80, "bold"),
                                  bg=P["card"], fg=P["yellow"],
                                  width=3)
        self.lbl_digit.pack()
        self.lbl_conf = tk.Label(res_card, text="",
                                 font=("Courier New", 9),
                                 bg=P["card"], fg=P["dim"])
        self.lbl_conf.pack()

        # Logit bar chart
        tk.Label(right, text="LOGIT SCORES  (raw output)",
                 font=("Courier New", 8, "bold"),
                 bg=P["bg"], fg=P["dim"]).pack(anchor="w", pady=(14, 4))

        self.bar_frame = tk.Frame(right, bg=P["bg"])
        self.bar_frame.pack(fill=tk.X)
        self._build_bars()

        # Log line
        self.log_var = tk.StringVar(value="Draw a digit and press PREDICT")
        tk.Label(self.root, textvariable=self.log_var,
                 font=("Courier New", 8),
                 bg="#0a0a10", fg=P["dim"],
                 anchor="w", padx=16, pady=5
                 ).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_bars(self):
        P = self.P
        self.bar_rows = []
        for i in range(10):
            row = tk.Frame(self.bar_frame, bg=P["bg"])
            row.pack(fill=tk.X, pady=2)

            lbl_digit = tk.Label(row, text=str(i), width=2,
                                 font=("Courier New", 9, "bold"),
                                 bg=P["bg"], fg=self.DIGIT_COLORS[i],
                                 anchor="e")
            lbl_digit.pack(side=tk.LEFT, padx=(0, 6))

            bar_bg = tk.Frame(row, bg=P["bar_bg"], height=16)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=P["dim"], height=16)
            bar_fill.place(x=0, y=0, width=0, relheight=1)

            lbl_val = tk.Label(row, text="", width=8,
                               font=("Courier New", 8),
                               bg=P["bg"], fg=P["dim"], anchor="w")
            lbl_val.pack(side=tk.LEFT, padx=(6, 0))

            self.bar_rows.append((bar_bg, bar_fill, lbl_val, lbl_digit))

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _start(self, e):
        self.drawing = True; self.lx = e.x; self.ly = e.y
        r = self.pen_w // 2
        self.canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r,
                                fill="white", outline="white")
        self.pil_draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=255)

    def _move(self, e):
        if not self.drawing: return
        self.canvas.create_line(self.lx, self.ly, e.x, e.y,
                                fill="white", width=self.pen_w,
                                capstyle=tk.ROUND, smooth=True)
        self.pil_draw.line([(self.lx, self.ly), (e.x, e.y)],
                           fill=255, width=self.pen_w)
        self.lx, self.ly = e.x, e.y

    def _end(self, e):
        self.drawing = False
        self._update_preview()

    def _update_preview(self, pixels=None):
        """Show the 28×28 image the model actually sees.
        If pixels is provided (preprocessed array), show that;
        otherwise fall back to a naive resize of the drawing."""
        if pixels is not None:
            img28_arr = pixels.reshape(28, 28).astype(np.float32)
            # Scale from [0,127] back to [0,255] for display
            img28_arr = np.clip(img28_arr * 2, 0, 255).astype(np.uint8)
        else:
            img28 = self.pil_img.resize((28, 28), Image.Resampling.LANCZOS)
            img28_arr = np.array(img28, dtype=np.uint8)

        big = np.repeat(np.repeat(img28_arr, 5, axis=0), 5, axis=1)  # 140×140
        self._tk_preview = tk.PhotoImage(width=140, height=140)
        rows = []
        for y in range(140):
            row_data = " ".join(
                "#{:02x}{:02x}{:02x}".format(v, v, v) for v in big[y]
            )
            rows.append("{" + row_data + "}")
        self._tk_preview.put(" ".join(rows))
        self.preview_canvas.create_image(0, 0, anchor="nw",
                                         image=self._tk_preview)

    def _clear(self):
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self.pil_img  = Image.new('L', (self.CW, self.CH), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.lbl_digit.config(text="—", fg=self.P["yellow"])
        self.lbl_conf.config(text="")
        self._reset_bars()
        self._log("Cleared — draw a new digit")

    def _reset_bars(self):
        for bar_bg, bar_fill, lbl_val, lbl_digit in self.bar_rows:
            bar_fill.place(width=0)
            lbl_val.config(text="", fg=self.P["dim"])

    # ── Inference ─────────────────────────────────────────────────────────────

    def _predict(self):
        if not self.mlp._loaded:
            self._log("Weights not loaded yet — please wait"); return
        self._log("Running inference…")
        threading.Thread(target=self._run_infer, daemon=True).start()

    def _run_infer(self):
        pixels = preprocess(self.pil_img)
        digit, logits = self.mlp.infer(pixels)
        self.root.after(0, self._show_result, digit, logits, pixels)

    def _show_result(self, digit, logits, pixels):
        if digit is None:
            self._log("Inference failed — check weight files"); return

        P = self.P
        color = self.DIGIT_COLORS[digit % 10]
        self.lbl_digit.config(text=str(digit), fg=color)

        # Softmax for confidence display
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        probs = exp / exp.sum()
        conf = probs[digit] * 100
        self.lbl_conf.config(
            text=f"Confidence  {conf:.1f}%   |   logit {logits[digit]:.1f}",
            fg=color)

        # Update bars
        lo, hi = logits.min(), logits.max()
        span = max(hi - lo, 1.0)
        for i, (bar_bg, bar_fill, lbl_val, _) in enumerate(self.bar_rows):
            W = bar_bg.winfo_width()
            if W < 4: W = 200
            frac = max((logits[i] - lo) / span, 0.0)
            fw = int(frac * W)
            bar_fill.config(bg=self.DIGIT_COLORS[i] if i == digit else P["dim"])
            bar_fill.place(width=fw)
            lbl_val.config(
                text=f"{logits[i]:+.1f}  {probs[i]*100:4.1f}%",
                fg=self.DIGIT_COLORS[i] if i == digit else P["dim"])

        # Update preview to show what model actually saw (centered+normalized)
        self._update_preview(pixels)
        active = sum(1 for p in pixels if p > 0)
        self._log(
            f"Predicted: {digit}  |  confidence: {conf:.1f}%  |  "
            f"active pixels: {active}/784"
        )

    # ── Weight Loading ────────────────────────────────────────────────────────

    def _load_weights(self):
        def _load():
            self.mlp.load()
            self.root.after(0, self._on_loaded)
        threading.Thread(target=_load, daemon=True).start()

    def _on_loaded(self):
        ok = self.mlp._loaded
        color = self.P["accent2"] if ok else self.P["err"]
        self.lbl_status.config(text=self.mlp.status, fg=color)
        self._log(self.mlp.status if ok else f"ERROR: {self.mlp.status}")

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _log(self, msg):
        self.log_var.set(msg)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()
