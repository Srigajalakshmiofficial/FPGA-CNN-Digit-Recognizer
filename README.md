# 🔢 MNIST Handwritten Digit Recognizer — FPGA + Python

> **Mini Project**
> A hardware-accelerated handwritten digit classifier implemented on an Altera DE2 FPGA using a quantized 4-layer Multilayer Perceptron (MLP), with Python-based GUI clients for both hardware and software inference.

---

## Overview

This project implements a real-time MNIST digit recognition system in which a **784→128→64→32→10 INT4 MLP** is synthesized onto a **Cyclone II FPGA (EP2C35F672C6)**. A user draws a digit on a PC GUI, which sends the 28×28 pixel image over UART to the FPGA, and the FPGA returns the predicted digit (0–9) in milliseconds.

A pure-Python software simulator is also included, which mirrors the Verilog hardware behavior exactly — useful for testing and validation without physical hardware.

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PC (Host)                                │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  fpga_client_28.py  │    │    digit_predictor_gui.py       │ │
│  │  (Hardware Mode)    │    │    (Software Simulator Mode)    │ │
│  │  Draw → UART → FPGA │    │    Draw → INT4 MLP in Python    │ │
│  └────────┬────────────┘    └─────────────────────────────────┘ │
└───────────┼─────────────────────────────────────────────────────┘
            │ UART @ 115200 baud  (786 bytes per image)
            │ Protocol: 0xFF + 784 bytes + 1 checksum
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Altera DE2 — Cyclone II FPGA                   │
│                                                                 │
│   uart_rx → pixel_buffer → mlp_infer → uart_tx                  │
│                                 │                               │
│                         784→128→64→32→10                        │
│                         INT4 Weights  INT16 Biases              │
│                         ReLU  [0,127]  Activations              │
│                                 │                               │
│              Result shown on HEX0 display + LEDR[17:14]         │
└─────────────────────────────────────────────────────────────────┘
```

---

##  Neural Network Details

| Property            | Value                              |
|---------------------|------------------------------------|
| Architecture        | 784 → 128 → 64 → 32 → 10          |
| Weight Precision    | INT4 (4-bit signed, packed 2/byte) |
| Bias Precision      | INT16 (16-bit signed)              |
| Activation Function | ReLU, clamped to [0, 127]          |
| Input               | 28×28 grayscale pixels → [0, 127] |
| Total MACs          | 110,912                            |
| Pre-trained model   | `mlp28_99acc.pt` (PyTorch)        |
| Reported Accuracy   | ~99% on MNIST test set             |

### Layer Dimensions

| Layer | Input | Output | Weights (INT4 bytes) | Biases |
|-------|-------|--------|----------------------|--------|
| FC1   | 784   | 128    | 50,176               | 128    |
| FC2   | 128   | 64     | 4,096                | 64     |
| FC3   | 64    | 32     | 1,024                | 32     |
| FC4   | 32    | 10     | 160                  | 10     |

---

## 🔌 Hardware

| Component              | Details                              |
|------------------------|--------------------------------------|
| FPGA Board             | Altera DE2                           |
| Device                 | Cyclone II EP2C35F672C6              |
| Clock                  | 50 MHz (`CLOCK_50`)                  |
| EDA Tool               | Quartus II 13.0 SP1 (Web Edition)    |
| UART                   | 8N1 @ 115,200 baud                  |
| UART RX Pin            | `GPIO_1[0]` ← HW597 TX              |
| UART TX Pin            | `GPIO_1[1]` → HW597 RX              |
| Reset                  | `KEY0` (active-low)                  |
| Result Display         | `HEX0` (7-segment) + `LEDR[17:14]` |

### LED Indicators

| Signal          | LED            |
|-----------------|----------------|
| Buffer ready    | `LEDG[0]`      |
| Checksum OK     | `LEDG[1]`      |
| Inference done  | `LEDG[2]`      |
| Predicted digit | `LEDR[17:14]`  |

---

## 📁 Project Structure

```
mlp_digit_try1/
│
├── mnist_top.v              # Top-level Verilog: UART RX/TX + pixel buffer + MLP inference
├── mnist_top.qpf            # Quartus II project file
├── mnist_top.qsf            # Quartus pin assignments & device settings
├── mnist_top.sof            # FPGA configuration bitstream (SRAM Object File)
├── mnist_top.pof            # FPGA programming file (Programmer Object File)
│
├── digit_predictor_gui.py   # Python GUI — software-only INT4 MLP simulator
├── fpga_client_28.py        # Python GUI — sends images to FPGA over UART
├── mlp_simulator.py         # Core INT4 MLP inference engine (pure Python/NumPy)
├── mlp28_99acc.pt           # Pre-trained PyTorch model (~99% accuracy)
│
├── fc1_weights.hex          # INT4 packed weights, FC1 (784×128)
├── fc1_bias.hex             # INT16 biases, FC1 (128 entries)
├── fc2_weights.hex          # INT4 packed weights, FC2 (128×64)
├── fc2_bias.hex             # INT16 biases, FC2 (64 entries)
├── fc3_weights.hex          # INT4 packed weights, FC3 (64×32)
├── fc3_bias.hex             # INT16 biases, FC3 (32 entries)
├── fc4_weights.hex          # INT4 packed weights, FC4 (32×10)
├── fc4_bias.hex             # INT16 biases, FC4 (10 entries)
│
└── quartus_hex_28_99/       # Hex files used by the GUI simulator
    ├── fc1_weights.hex
    ├── fc1_bias.hex
    ├── fc2_weights.hex
    ├── fc2_bias.hex
    ├── fc3_weights.hex
    ├── fc3_bias.hex
    ├── fc4_weights.hex
    └── fc4_bias.hex
```

---

##  Getting Started

### Prerequisites

- **Python 3.8+**
- Install Python dependencies:

```bash
pip install pillow numpy pyserial
```

- **Hardware mode only:** Altera Quartus II 13.0 SP1, a programmed DE2 board, and a USB-UART adapter (HW597 / CH340 / FTDI / CP210x)

---

### Mode 1: Software Simulator (No FPGA Required)

Run the GUI that uses the bundled hex files to perform INT4 inference entirely in Python — mirrors the Verilog behavior exactly:

```bash
python digit_predictor_gui.py
```

- Draw a digit (0–9) on the canvas
- Click **Predict** to run inference
- The predicted digit and logits bar chart are shown instantly

> Hex files are loaded from `./quartus_hex_28_99/`

---

### Mode 2: FPGA Hardware Inference

1. **Program the FPGA** using Quartus Programmer with `mnist_top.sof` (volatile) or `mnist_top.pof` (non-volatile).

2. **Connect** the DE2 board to your PC via a USB-UART adapter:
   - `GPIO_1[0]` ← UART Adapter TX
   - `GPIO_1[1]` → UART Adapter RX
   - GND shared

3. **Launch the client:**

```bash
python fpga_client_28.py
```

4. Select the correct COM port (auto-detected if using FTDI/CH340/CP210x), then click **CONNECT**.

5. Draw a digit and click **▶ SEND TO FPGA**. The prediction is returned and displayed.

---

##  Communication Protocol

| Direction | Content                               | Bytes   |
|-----------|---------------------------------------|---------|
| PC → FPGA | `0xFF` (sync byte)                    | 1       |
| PC → FPGA | 784 pixel bytes (0–127, row-major)    | 784     |
| PC → FPGA | Checksum (sum of 784 bytes, mod 256)  | 1       |
| **Total** |                                       | **786** |
| FPGA → PC | `0x00`–`0x09` (predicted digit)       | 1       |
| FPGA → PC | `0xFF` (checksum error indicator)     | 1       |

---

##  Performance

| Metric         | Value                                   |
|----------------|-----------------------------------------|
| Clock          | 50 MHz                                  |
| UART Transfer  | 786 bytes @ 115,200 baud ≈ **68 ms**   |
| MLP Inference  | ~111,000 cycles ≈ **2.2 ms** @ 50 MHz |
| Total Latency  | ~70 ms end-to-end                       |

---

##  Building from Source (Quartus II)

1. Open `mnist_top.qpf` in **Quartus II 13.0 SP1**.
2. Ensure `mnist_top.v` and all `.hex` files are in the project root.
3. Run **Processing → Start Compilation** (Ctrl+L).
4. Use the **Programmer** (Tools → Programmer) to flash `mnist_top.sof` to the DE2 board.

> ⚠️ The project uses **Verilog-1995** syntax (Quartus II 13.0 limitation). Do not use SystemVerilog constructs.

---

## Weight Format (INT4 Packed Hex)

Weights are stored in Quartus-compatible Intel HEX format, with **two INT4 values packed per byte**:

```
Byte value:  [HI nibble | LO nibble]
              odd index    even index
```

- Each nibble is a **4-bit signed integer** (two's complement, range −8 to +7).
- Flat index: `flat = neuron_index * n_cols + input_index`

Biases are **INT16** (16-bit signed), stored as one 4-hex-char value per line.

---

## 📊 Verilog Module Hierarchy

```
mnist_top
├── uart_rx          # 8N1 UART receiver with dual-FF metastability sync
├── pixel_buffer     # Collects 786-byte frame, validates checksum
├── mlp_infer        # 4-layer INT4 MLP inference engine
│   ├── FC1 layer    # 784→128, ROM-backed weights, ReLU
│   ├── FC2 layer    # 128→64,  ROM-backed weights, ReLU
│   ├── FC3 layer    # 64→32,   ROM-backed weights, ReLU
│   └── FC4 layer    # 32→10,   ROM-backed weights, argmax
├── uart_tx          # 8N1 UART transmitter
└── seg7             # 7-segment decoder for HEX0 display
```

---

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) — LeCun et al.
- [Altera DE2 User Manual](https://www.intel.com/content/www/us/en/programmable/support/training/university/materials.html)
- [Quartus II Handbook](https://www.intel.com/content/www/us/en/programmable/documentation/lit-index.html)
- PyTorch — for training `mlp28_99acc.pt`

---

## License

This project was developed as a Mini Project for academic purposes.

---

*Where silicon meets intelligence — digit by digit.*
