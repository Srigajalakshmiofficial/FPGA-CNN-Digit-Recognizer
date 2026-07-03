// ============================================================
//  mnist_top_4layer.v  —  784→128→64→32→10 INT4 MLP
//  DE2 / Cyclone II EP2C35F672C6
//  Quartus II Web Edition 13.0 (Verilog-1995 only)
//
//  GPIO_1[0] = UART RX  (HW597 TX)
//  GPIO_1[1] = UART TX  (HW597 RX)
//  KEY0      = reset (active-low)
//
//  Protocol PC→FPGA : 0xFF + 784 bytes + 1 checksum  (786 total)
//  Protocol FPGA→PC : 1 byte (0x00–0x09) or 0xFF on error
//
//  Hex files (in Quartus project folder):
//    fc1_weights.hex  50176 lines  (784×128/2)
//    fc2_weights.hex   4096 lines  (128×64/2)
//    fc3_weights.hex   1024 lines  ( 64×32/2)
//    fc4_weights.hex    160 lines  ( 32×10/2)
//    fc1_bias.hex       128 entries
//    fc2_bias.hex        64 entries
//    fc3_bias.hex        32 entries
//    fc4_bias.hex        10 entries
//
//  Total MACs  : 784×128 + 128×64 + 64×32 + 32×10
//              = 100352 + 8192 + 2048 + 320 = 110912
//  Inference   : ~111K cycles ≈ 2
//  UART RX     : 786 bytes @ 115200 ≈ 68 ms
// =================.2 ms @ 50 MHz===========================================

module mnist_top (
    input  wire        CLOCK_50,
    input  wire        KEY0,
    inout  wire [35:0] GPIO_1,
    output wire [17:0] LEDR,
    output wire [ 8:0] LEDG,
    output wire [ 6:0] HEX0
);

    wire rst_n    = KEY0;
    wire uart_rx  = GPIO_1[0];
    wire uart_tx_w;
    assign GPIO_1[1] = uart_tx_w;

    wire [7:0] rx_byte;
    wire       rx_valid;

    uart_rx #(.CLK_HZ(50_000_000), .BAUD(115_200)) u_rx (
        .clk   (CLOCK_50), .rst_n (rst_n),
        .rx    (uart_rx),  .dout  (rx_byte), .valid (rx_valid)
    );

    wire [6271:0] pixel_bus;
    wire          buf_ready;
    wire          chksum_ok;

    pixel_buffer u_buf (
        .clk      (CLOCK_50), .rst_n    (rst_n),
        .rx_byte  (rx_byte),  .rx_valid (rx_valid),
        .pixels   (pixel_bus),.ready    (buf_ready), .chksum_ok(chksum_ok)
    );

    wire [3:0] pred_digit;
    wire       pred_done;

    mlp_infer u_mlp (
        .clk    (CLOCK_50), .rst_n  (rst_n),
        .start  (buf_ready & chksum_ok),
        .pixels (pixel_bus),
        .result (pred_digit), .done (pred_done)
    );

    wire [7:0] tx_byte = chksum_ok ? {4'h0, pred_digit} : 8'hFF;

    uart_tx #(.CLK_HZ(50_000_000), .BAUD(115_200)) u_tx (
        .clk  (CLOCK_50), .rst_n (rst_n),
        .din  (tx_byte),
        .send (pred_done | (buf_ready & ~chksum_ok)),
        .tx   (uart_tx_w), .busy ()
    );

    reg [3:0] last_result;
    always @(posedge CLOCK_50 or negedge rst_n) begin
        if (!rst_n)         last_result <= 4'hF;
        else if (pred_done) last_result <= pred_digit;
    end

    assign LEDR[17:14] = last_result;
    assign LEDR[13:0]  = 14'd0;
    assign LEDG[0]     = buf_ready;
    assign LEDG[1]     = chksum_ok;
    assign LEDG[2]     = pred_done;
    assign LEDG[8:3]   = 6'd0;

    seg7 u_seg (.digit(last_result), .seg(HEX0));

endmodule


// ============================================================
//  uart_rx — 8N1, dual-FF metastability sync
// ============================================================
module uart_rx #(
    parameter CLK_HZ = 50_000_000,
    parameter BAUD   = 115_200
)(
    input  wire       clk, rst_n, rx,
    output reg  [7:0] dout,
    output reg        valid
);
    localparam FULL = CLK_HZ / BAUD;
    localparam HALF = FULL / 2;
    localparam IDLE = 2'd0, START = 2'd1, DATA = 2'd2, STOP = 2'd3;

    reg [1:0] state;
    reg [9:0] cnt;
    reg [2:0] bit_idx;
    reg [7:0] shift;
    reg       s0, s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin s0<=1; s1<=1; end
        else        begin s0<=rx; s1<=s0; end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE; valid<=0; dout<=0; cnt<=0; bit_idx<=0; shift<=0;
        end else begin
            valid <= 0;
            case (state)
                IDLE:  if (!s1) begin state<=START; cnt<=HALF[9:0]; end

                START: if (cnt==0) begin
                           if (!s1) begin state<=DATA; cnt<=FULL[9:0]; bit_idx<=0; end
                           else          state<=IDLE;
                       end else cnt<=cnt-10'd1;

                DATA:  if (cnt==0) begin
                           shift <= {s1, shift[7:1]};
                           cnt   <= FULL[9:0];
                           if (bit_idx==3'd7) state<=STOP;
                           else               bit_idx<=bit_idx+3'd1;
                       end else cnt<=cnt-10'd1;

                STOP:  if (cnt==0) begin
                           dout<=shift; valid<=1; state<=IDLE;
                       end else cnt<=cnt-10'd1;
            endcase
        end
    end
endmodule


// ============================================================
//  uart_tx — 8N1
// ============================================================
module uart_tx #(
    parameter CLK_HZ = 50_000_000,
    parameter BAUD   = 115_200
)(
    input  wire       clk, rst_n,
    input  wire [7:0] din,
    input  wire       send,
    output reg        tx,
    output reg        busy
);
    localparam FULL = CLK_HZ / BAUD;
    reg [9:0] cnt;
    reg [3:0] bit_idx;
    reg [9:0] frame;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tx<=1; busy<=0; cnt<=0; bit_idx<=0; frame<=0; end
        else if (!busy && send) begin
            frame<=  {1'b1, din, 1'b0}; tx<=0;
            busy<=1; bit_idx<=0; cnt<=FULL[9:0]-10'd1;
        end else if (busy) begin
            if (cnt==0) begin
                cnt <= FULL[9:0]-10'd1;
                if (bit_idx==4'd9) begin tx<=1; busy<=0; end
                else               begin tx<=frame[bit_idx]; bit_idx<=bit_idx+4'd1; end
            end else cnt<=cnt-10'd1;
        end
    end
endmodule


// ============================================================
//  pixel_buffer — 28×28 = 784 bytes
//  Protocol: 0xFF marker + 784 bytes + 1 checksum
// ============================================================
module pixel_buffer (
    input  wire          clk, rst_n,
    input  wire [7:0]    rx_byte,
    input  wire          rx_valid,
    output reg [6271:0]  pixels,
    output reg           ready,
    output reg           chksum_ok
);
    localparam WAIT = 2'd0, RECV = 2'd1, CHKSUM = 2'd2;

    reg [1:0] state;
    reg [9:0] idx;
    reg [7:0] rsum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=WAIT; ready<=0; chksum_ok<=0; idx<=0; rsum<=0;
        end else begin
            ready <= 0;
            if (rx_valid) begin
                case (state)
                    WAIT: if (rx_byte==8'hFF) begin state<=RECV; idx<=0; rsum<=0; end

                    RECV: begin
                        pixels[6271 - idx*8 -: 8] <= rx_byte;
                        rsum <= rsum + rx_byte;
                        if (idx==10'd783) state<=CHKSUM;
                        else              idx<=idx+10'd1;
                    end

                    CHKSUM: begin
                        chksum_ok <= (rx_byte==rsum);
                        ready     <= 1;
                        state     <= WAIT;
                    end
                endcase
            end
        end
    end
endmodule


// ============================================================
//  mlp_infer — 784→128→64→32→10  INT4
//  Sequential MAC: 1 multiply-accumulate per clock cycle
// ============================================================
module mlp_infer (
    input  wire          clk, rst_n,
    input  wire          start,
    input  wire [6271:0] pixels,
    output reg  [3:0]    result,
    output reg           done
);
    // ── FSM ──────────────────────────────────────────────────
    localparam S_IDLE   = 3'd0;
    localparam S_L1     = 3'd1;
    localparam S_L2     = 3'd2;
    localparam S_L3     = 3'd3;
    localparam S_L4     = 3'd4;

    reg [2:0]  state;
    reg [16:0] neuron_idx;
    reg [9:0]  input_idx;
    reg signed [22:0] acc;

    // ── store_neuron flag: set on last-MAC cycle, processed next cycle ──
    reg store_neuron;

    // ── Activation buffers ───────────────────────────────────
    reg signed [7:0] a1 [0:127];
    reg signed [7:0] a2 [0:63];
    reg signed [7:0] a3 [0:31];
    reg signed [15:0] logit [0:9];

    // ── Weight ROMs ──────────────────────────────────────────
    reg [7:0] fc1_w [0:50175];
    reg [7:0] fc2_w [0:4095];
    reg [7:0] fc3_w [0:1023];
    reg [7:0] fc4_w [0:159];

    reg signed [15:0] fc1_b [0:127];
    reg signed [15:0] fc2_b [0:63];
    reg signed [15:0] fc3_b [0:31];
    reg signed [15:0] fc4_b [0:9];

    initial begin
        $readmemh("fc1_weights.hex", fc1_w);
        $readmemh("fc2_weights.hex", fc2_w);
        $readmemh("fc3_weights.hex", fc3_w);
        $readmemh("fc4_weights.hex", fc4_w);
        $readmemh("fc1_bias.hex",    fc1_b);
        $readmemh("fc2_bias.hex",    fc2_b);
        $readmemh("fc3_bias.hex",    fc3_b);
        $readmemh("fc4_bias.hex",    fc4_b);
    end

    // ── MAC wires ────────────────────────────────────────────
    reg signed [3:0]  w_val;
    reg signed [7:0]  x_val;
    reg signed [10:0] prod;
    reg [7:0]  rom_byte;

    // ── Argmax ───────────────────────────────────────────────
    reg signed [15:0] max_val;
    reg [3:0]  max_idx;
    reg [3:0]  am_cnt;
    reg        do_argmax;

    integer k, flat;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=0; neuron_idx<=0; input_idx<=0; acc<=0;
            store_neuron<=0;
            for (k=0;k<128;k=k+1) a1[k]<=0;
            for (k=0;k<64; k=k+1) a2[k]<=0;
            for (k=0;k<32; k=k+1) a3[k]<=0;
            for (k=0;k<10; k=k+1) logit[k]<=0;
            result<=0; max_val<=0; max_idx<=0; am_cnt<=0; do_argmax<=0;
        end else begin
            // ── Inline argmax ────────────────────────────────
            if (do_argmax) begin
                if (am_cnt == 0) begin
                    max_val <= logit[0]; max_idx <= 0; am_cnt <= 1;
                end else begin
                    if ($signed(logit[am_cnt]) > $signed(max_val)) begin
                        max_val <= logit[am_cnt]; max_idx <= am_cnt;
                    end
                    if (am_cnt == 4'd9) begin
                        result    <= max_idx;
                        done      <= 1;
                        do_argmax <= 0;
                    end else am_cnt <= am_cnt + 4'd1;
                end
            end else done <= 0;

            case (state)

            S_IDLE: begin
                store_neuron <= 0;
                if (start && !do_argmax) begin
                    state      <= S_L1;
                    neuron_idx <= 0; input_idx <= 0;
                    acc <= {{7{fc1_b[0][15]}}, fc1_b[0]};
                end
            end

            // ── FC1: 784 × 128 ───────────────────────────────
            S_L1: begin
                if (store_neuron) begin
                    store_neuron <= 0;
                    if      (acc <= 0)   a1[neuron_idx[6:0]] <= 8'sd0;
                    else if (acc > 127)  a1[neuron_idx[6:0]] <= 8'sd127;
                    else                 a1[neuron_idx[6:0]] <= acc[7:0];
                    if (neuron_idx == 17'd127) begin
                        state <= S_L2; neuron_idx <= 0; input_idx <= 0;
                        acc <= {{7{fc2_b[0][15]}}, fc2_b[0]};
                    end else begin
                        neuron_idx <= neuron_idx + 17'd1;
                        acc <= {{7{fc1_b[neuron_idx+1][15]}}, fc1_b[neuron_idx+1]};
                    end
                end else begin
                    flat     = neuron_idx * 784 + input_idx;
                    rom_byte = fc1_w[flat[16:1]];
                    w_val    = flat[0] ? $signed(rom_byte[7:4]) : $signed(rom_byte[3:0]);
                    x_val    = pixels[6271 - input_idx*8 -: 8];
                    prod     = $signed(w_val) * $signed(x_val);
                    acc     <= acc + {{12{prod[10]}}, prod};
                    if (input_idx == 10'd783) begin
                        input_idx    <= 0;
                        store_neuron <= 1;
                    end else input_idx <= input_idx + 10'd1;
                end
            end

            // ── FC2: 128 × 64 ────────────────────────────────
            S_L2: begin
                if (store_neuron) begin
                    store_neuron <= 0;
                    if      (acc <= 0)   a2[neuron_idx[5:0]] <= 8'sd0;
                    else if (acc > 127)  a2[neuron_idx[5:0]] <= 8'sd127;
                    else                 a2[neuron_idx[5:0]] <= acc[7:0];
                    if (neuron_idx == 17'd63) begin
                        state <= S_L3; neuron_idx <= 0; input_idx <= 0;
                        acc <= {{7{fc3_b[0][15]}}, fc3_b[0]};
                    end else begin
                        neuron_idx <= neuron_idx + 17'd1;
                        acc <= {{7{fc2_b[neuron_idx+1][15]}}, fc2_b[neuron_idx+1]};
                    end
                end else begin
                    flat     = neuron_idx * 128 + input_idx;
                    rom_byte = fc2_w[flat[12:1]];
                    w_val    = flat[0] ? $signed(rom_byte[7:4]) : $signed(rom_byte[3:0]);
                    x_val    = a1[input_idx[6:0]];
                    prod     = $signed(w_val) * $signed(x_val);
                    acc     <= acc + {{12{prod[10]}}, prod};
                    if (input_idx == 10'd127) begin
                        input_idx    <= 0;
                        store_neuron <= 1;
                    end else input_idx <= input_idx + 10'd1;
                end
            end

            // ── FC3: 64 × 32 ─────────────────────────────────
            S_L3: begin
                if (store_neuron) begin
                    store_neuron <= 0;
                    if      (acc <= 0)   a3[neuron_idx[4:0]] <= 8'sd0;
                    else if (acc > 127)  a3[neuron_idx[4:0]] <= 8'sd127;
                    else                 a3[neuron_idx[4:0]] <= acc[7:0];
                    if (neuron_idx == 17'd31) begin
                        state <= S_L4; neuron_idx <= 0; input_idx <= 0;
                        acc <= {{7{fc4_b[0][15]}}, fc4_b[0]};
                    end else begin
                        neuron_idx <= neuron_idx + 17'd1;
                        acc <= {{7{fc3_b[neuron_idx+1][15]}}, fc3_b[neuron_idx+1]};
                    end
                end else begin
                    flat     = neuron_idx * 64 + input_idx;
                    rom_byte = fc3_w[flat[11:1]];
                    w_val    = flat[0] ? $signed(rom_byte[7:4]) : $signed(rom_byte[3:0]);
                    x_val    = a2[input_idx[5:0]];
                    prod     = $signed(w_val) * $signed(x_val);
                    acc     <= acc + {{12{prod[10]}}, prod};
                    if (input_idx == 10'd63) begin
                        input_idx    <= 0;
                        store_neuron <= 1;
                    end else input_idx <= input_idx + 10'd1;
                end
            end

            // ── FC4: 32 × 10 ─────────────────────────────────
            S_L4: begin
                if (store_neuron) begin
                    store_neuron <= 0;
                    // acc is fully accumulated — safe to latch logit
                    logit[neuron_idx[3:0]] <= acc[15:0];
                    if (neuron_idx == 17'd9) begin
                        do_argmax  <= 1;
                        am_cnt     <= 0;
                        neuron_idx <= 0;
                        state      <= S_IDLE;
                    end else begin
                        neuron_idx <= neuron_idx + 17'd1;
                        acc <= {{7{fc4_b[neuron_idx+1][15]}}, fc4_b[neuron_idx+1]};
                    end
                end else begin
                    flat     = neuron_idx * 32 + input_idx;
                    rom_byte = fc4_w[flat[8:1]];
                    w_val    = flat[0] ? $signed(rom_byte[7:4]) : $signed(rom_byte[3:0]);
                    x_val    = a3[input_idx[4:0]];
                    prod     = $signed(w_val) * $signed(x_val);
                    acc     <= acc + {{12{prod[10]}}, prod};
                    if (input_idx == 10'd31) begin
                        input_idx    <= 0;
                        store_neuron <= 1;
                    end else input_idx <= input_idx + 10'd1;
                end
            end

            endcase
        end
    end
endmodule


// ============================================================
//  seg7 — active-low, DE2 standard
// ============================================================
module seg7 (
    input  wire [3:0] digit,
    output reg  [6:0] seg
);
    always @(*) begin
        case (digit)
            4'd0: seg = 7'b1000000;
            4'd1: seg = 7'b1111001;
            4'd2: seg = 7'b0100100;
            4'd3: seg = 7'b0110000;
            4'd4: seg = 7'b0011001;
            4'd5: seg = 7'b0010010;
            4'd6: seg = 7'b0000010;
            4'd7: seg = 7'b1111000;
            4'd8: seg = 7'b0000000;
            4'd9: seg = 7'b0010000;
            default: seg = 7'b1111111;
        endcase
    end
endmodule
