// Decode the FP32 accumulator input into sign, 24-bit significand and a
// signed unbiased exponent. Subnormals are kept.
module tc_unpack32 #(
    parameter int EW = 12
) (
    input  logic [31:0]   x,
    output logic          sign,
    output logic [23:0]   sig,
    output logic [EW-1:0] exp,
    output logic          is_zero,
    output logic          is_inf,
    output logic          is_nan
);
    logic ef_zero, ef_max;
    always_comb begin
        sign    = x[31];
        ef_zero = (x[30:23] == 8'h00);
        ef_max  = (x[30:23] == 8'hFF);
        is_nan  = ef_max && (x[22:0] != 23'd0);
        is_inf  = ef_max && (x[22:0] == 23'd0);
        is_zero = ef_zero && (x[22:0] == 23'd0);
        sig     = {~ef_zero, x[22:0]};
        exp     = {{(EW-8){1'b0}}, (ef_zero ? 8'd1 : x[30:23])} - EW'(127);
    end
endmodule
