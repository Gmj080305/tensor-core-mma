// Decode an FP16 or BF16 operand into sign, 11-bit significand (10 fractional
// bits, BF16 widened with three zero bits) and a signed unbiased exponent.
// Subnormals are kept (exponent = emin, hidden bit = 0).
module tc_unpack16 #(
    parameter int EW = 12              // signed exponent width
) (
    input  logic          fmt,         // 0: FP16, 1: BF16
    input  logic [15:0]   x,
    output logic          sign,
    output logic [10:0]   sig,
    output logic [EW-1:0] exp,         // signed
    output logic          is_zero,
    output logic          is_inf,
    output logic          is_nan
);
    logic [7:0] ef;     // exponent field (zero-extended for FP16)
    logic [9:0] fr;     // fraction field, left-aligned to 10 bits
    logic       ef_zero, ef_max;

    always_comb begin
        sign = x[15];
        if (!fmt) begin
            ef     = {3'b000, x[14:10]};
            fr     = x[9:0];
            ef_max = (x[14:10] == 5'h1F);
        end else begin
            ef     = x[14:7];
            fr     = {x[6:0], 3'b000};
            ef_max = (x[14:7] == 8'hFF);
        end
        ef_zero = (ef == 8'd0);
        is_nan  = ef_max && (fr != 10'd0);
        is_inf  = ef_max && (fr == 10'd0);
        is_zero = ef_zero && (fr == 10'd0);
        sig     = {~ef_zero, fr};
        // unbiased exponent; a subnormal uses exponent field 1 (= emin).
        // Written without signed casts so that every tool agrees (two's complement).
        exp     = {{(EW-8){1'b0}}, (ef_zero ? 8'd1 : ef)} - (fmt ? EW'(127) : EW'(15));
    end
endmodule
