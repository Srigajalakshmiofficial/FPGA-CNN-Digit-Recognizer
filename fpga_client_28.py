"""
fpga_client_28.py
==================
PC client for 28×28 MNIST on DE2.
Protocol: 0xFF + 784 bytes + checksum  (786 total)
"""

import tkinter as tk
from tkinter import messagebox
import serial, serial.tools.list_ports
import threading, time
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
MEAN, STD    = 0.1307, 0.3081

class FPGAClient28:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 28×28 | DE2 Cyclone II")
        self.root.geometry("520x680")
        self.root.configure(bg="#0f0f0f")
        self.root.resizable(False, False)

        self.uart = None
        self.connected = False
        self.drawing = False
        self.CW = self.CH = 280
        self.pen_w = 18
        self.pil_img  = Image.new('L', (self.CW, self.CH), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

        P = {
            "bg":"#0f0f0f","panel":"#1a1a1a","accent":"#00d4aa",
            "err":"#ff3366","text":"#e8e8e8","dim":"#666",
            "canvas":"#050505","yellow":"#ffd460"
        }
        self.P = P

        # ── Header ──
        hdr = tk.Frame(root, bg=P["panel"], height=52)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr, text="▮ MNIST 28×28", font=("Courier New",13,"bold"),
                 bg=P["panel"], fg=P["accent"]).pack(side=tk.LEFT, padx=14)
        self.dot = tk.Canvas(hdr,width=10,height=10,bg=P["panel"],highlightthickness=0)
        self.dot.pack(side=tk.RIGHT,padx=8)
        self.dot.create_oval(2,2,9,9,fill=P["err"],tag="d")
        self.lbl_conn = tk.Label(hdr,text="Offline",font=("Courier New",9),
                                  bg=P["panel"],fg=P["err"])
        self.lbl_conn.pack(side=tk.RIGHT,padx=4)

        # ── Connection ──
        conn = tk.Frame(root,bg=P["bg"],pady=8); conn.pack(fill=tk.X,padx=16)
        for lbl,var,val,w in [("PORT","port_var",DEFAULT_PORT,9),
                               ("BAUD","baud_var",str(DEFAULT_BAUD),8)]:
            tk.Label(conn,text=lbl,font=("Courier New",8),bg=P["bg"],fg=P["dim"]).pack(side=tk.LEFT)
            v = tk.StringVar(value=val); setattr(self,var,v)
            tk.Entry(conn,textvariable=v,width=w,font=("Courier New",10),
                     bg=P["panel"],fg=P["text"],insertbackground=P["text"],
                     relief=tk.FLAT,highlightthickness=1,
                     highlightbackground="#2a2a2a").pack(side=tk.LEFT,padx=(4,12))
        self.btn_c = tk.Button(conn,text="CONNECT",command=self._toggle,
                               font=("Courier New",9,"bold"),bg=P["accent"],fg=P["bg"],
                               relief=tk.FLAT,padx=14,pady=4,cursor="hand2",bd=0)
        self.btn_c.pack(side=tk.LEFT)

        # ── Canvas ──
        tk.Label(root,text="DRAW DIGIT (0-9)",font=("Courier New",8,"bold"),
                 bg=P["bg"],fg=P["dim"]).pack(anchor="w",padx=16,pady=(8,4))
        outer = tk.Frame(root,bg=P["accent"],padx=1,pady=1); outer.pack()
        inner = tk.Frame(outer,bg="#2a2a2a",padx=2,pady=2); inner.pack()
        self.canvas = tk.Canvas(inner,width=self.CW,height=self.CH,
                                bg=P["canvas"],cursor="crosshair",highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",self._start)
        self.canvas.bind("<B1-Motion>",self._move)
        self.canvas.bind("<ButtonRelease-1>",self._end)

        # ── Result ──
        res = tk.Frame(root, bg=P["bg"], pady=10); res.pack(fill=tk.X, padx=16)
        tk.Label(res, text="RESULT", font=("Courier New", 8),
                 bg=P["bg"], fg=P["dim"]).pack(side=tk.LEFT)
        self.lbl_result = tk.Label(res, text="—",
                                   font=("Courier New", 14, "bold"),
                                   bg=P["bg"], fg=P["dim"])
        self.lbl_result.pack(side=tk.LEFT, padx=12)

        # ── Buttons ──
        btns = tk.Frame(root,bg=P["bg"]); btns.pack(pady=4)
        tk.Button(btns,text="CLEAR",command=self._clear,
                  font=("Courier New",10,"bold"),bg=P["panel"],fg=P["dim"],
                  relief=tk.FLAT,padx=24,pady=8,cursor="hand2",bd=0
                 ).pack(side=tk.LEFT,padx=(0,10))
        tk.Button(btns,text="▶  SEND TO FPGA",command=self._send,
                  font=("Courier New",10,"bold"),bg=P["accent"],fg=P["bg"],
                  relief=tk.FLAT,padx=24,pady=8,cursor="hand2",bd=0
                 ).pack(side=tk.LEFT)

        # ── Log ──
        self.log_var = tk.StringVar(value="Ready")
        tk.Label(root,textvariable=self.log_var,font=("Courier New",8),
                 bg="#111",fg=P["dim"],anchor="w",padx=14,pady=6
                ).pack(side=tk.BOTTOM,fill=tk.X)

        self._auto_detect()

    def _auto_detect(self):
        for p in serial.tools.list_ports.comports():
            if any(k in p.description.upper() for k in
                   ("FTDI","CH340","CP210","HW597","USB","BLASTER")):
                self.port_var.set(p.device)
                self._log(f"Auto-detected: {p.device}")
                return

    def _toggle(self):
        if self.connected: self._disconnect()
        else:              self._connect()

    def _connect(self):
        try:
            self.uart = serial.Serial(self.port_var.get().strip(),
                                      int(self.baud_var.get()), timeout=3.0)
            self.uart.reset_input_buffer()
            self.connected = True
            self._ui_conn(True)
            self._log(f"Connected {self.port_var.get()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _disconnect(self):
        if self.uart:
            try: self.uart.close()
            except: pass
        self.uart = None; self.connected = False
        self._ui_conn(False); self._log("Disconnected")

    def _ui_conn(self, ok):
        P = self.P
        c = P["accent"] if ok else P["err"]
        self.dot.delete("d"); self.dot.create_oval(2,2,9,9,fill=c,tag="d")
        self.lbl_conn.config(text=("Online" if ok else "Offline"), fg=c)
        self.btn_c.config(text=("DISCONNECT" if ok else "CONNECT"),
                          bg=(P["err"] if ok else P["accent"]))

    def _start(self, e):
        self.drawing = True; self.lx = e.x; self.ly = e.y
        r = self.pen_w//2
        self.canvas.create_oval(e.x-r,e.y-r,e.x+r,e.y+r,fill="white",outline="white")
        self.pil_draw.ellipse([e.x-r,e.y-r,e.x+r,e.y+r],fill=255)

    def _move(self, e):
        if not self.drawing: return
        self.canvas.create_line(self.lx,self.ly,e.x,e.y,fill="white",
                                width=self.pen_w,capstyle=tk.ROUND,smooth=True)
        self.pil_draw.line([(self.lx,self.ly),(e.x,e.y)],fill=255,width=self.pen_w)
        self.lx,self.ly = e.x,e.y

    def _end(self, e): self.drawing = False

    def _clear(self):
        self.canvas.delete("all")
        self.pil_img  = Image.new('L', (self.CW, self.CH), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.lbl_result.config(text="—", fg=self.P["dim"])
        self._log("Cleared")

    def _preprocess(self):
        # MNIST-style: crop to bounding box → 20×20 → center in 28×28 → [0,127]
        img = self.pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        arr = np.array(img, dtype=np.uint8)
        mask = arr > 20
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            return bytes(784)
        rmin, rmax = np.where(rows_any)[0][[0, -1]]
        cmin, cmax = np.where(cols_any)[0][[0, -1]]
        h = max(rmax - rmin + 1, 1); w = max(cmax - cmin + 1, 1)
        pr = max(int(h * 0.15), 2); pc = max(int(w * 0.15), 2)
        r0 = max(rmin-pr, 0); r1 = min(rmax+pr+1, arr.shape[0])
        c0 = max(cmin-pc, 0); c1 = min(cmax+pc+1, arr.shape[1])
        cropped = Image.fromarray(arr[r0:r1, c0:c1])
        cw, ch = cropped.size
        scale = 20 / max(cw, ch)
        nw = max(int(cw*scale), 1); nh = max(int(ch*scale), 1)
        resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (28, 28), 0)
        canvas.paste(resized, ((28-nw)//2, (28-nh)//2))
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.5))
        out = np.clip(np.array(canvas, dtype=np.float32) / 255.0 * 127, 0, 127).astype(np.uint8)
        return out.flatten().tobytes()

    def _send(self):
        if not self.connected:
            messagebox.showwarning("Not Connected","Connect first."); return
        pixels = self._preprocess()
        # Checksum: must match FPGA's 8-bit rsum register (wraps at 256)
        # Use numpy uint8 to guarantee identical 8-bit overflow behaviour.
        checksum = int(np.array(list(pixels), dtype=np.uint8).sum(dtype=np.uint8))
        packet   = bytes([0xFF]) + pixels + bytes([checksum])
        self._log(f"Sending {len(packet)} bytes  chk=0x{checksum:02X}...")
        try:
            # Clear BEFORE write to purge stale bytes from previous transaction
            self.uart.reset_input_buffer()
            self.uart.write(packet)
            self.uart.flush()
            # --- critical: clear again AFTER flush so any echo/stale reply
            # that arrived while we were transmitting doesn't poison the read.
            # Give the FPGA a tiny moment to begin its inference pipeline first.
            time.sleep(0.01)
            self.uart.reset_input_buffer()
        except Exception as e:
            self._log(f"Write error: {e}"); return
        t0 = time.perf_counter()
        threading.Thread(target=self._recv, args=(t0,), daemon=True).start()

    def _recv(self, t0):
        try:
            # Read 1 result byte from FPGA (0x00-0x09 = digit, 0xFF = chksum error)
            resp = self.uart.read(1)
            rtt  = (time.perf_counter() - t0) * 1000
            if not resp:
                self.root.after(0, self._log, "Timeout — check wiring/baud"); return
            b = resp[0]
            # Drain any unexpected extra bytes (e.g. FPGA debug output)
            leftover = self.uart.read(self.uart.in_waiting or 0)
            if leftover:
                self.root.after(0, self._log,
                    f"Raw: 0x{b:02X}  (+{len(leftover)} extra bytes drained)")
            if b == 0xFF:
                self.root.after(0, self._show_err, rtt)
            elif 0 <= b <= 9:
                self.root.after(0, self._show_ok, b, rtt)
            else:
                self.root.after(0, self._log, f"Unexpected byte: 0x{b:02X}")
        except Exception as e:
            self.root.after(0, self._log, f"Read error: {e}")

    def _show_ok(self, d, rtt):
        self.lbl_result.config(text="Predicted ✓", fg=self.P["accent"])
        self._log(f"Predicted: {d}  |  {rtt:.1f} ms")

    def _show_err(self, rtt):
        self.lbl_result.config(text="Not Predicted ✗", fg=self.P["err"])
        self._log("Not Predicted — checksum error, check wiring")

    def _log(self, m): self.log_var.set(m)

    def on_close(self):
        self._disconnect(); self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app  = FPGAClient28(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
