// Leading-zero counter. cnt = number of zeros above the most significant one;
// cnt = W when x == 0.
module tc_lzc #(
    parameter int W  = 32,
    parameter int CW = $clog2(W + 1)
) (
    input  logic [W-1:0]  x,
    output logic [CW-1:0] cnt
);
    always_comb begin
        cnt = CW'(W);
        for (int i = 0; i < W; i++) begin
            if (x[i]) cnt = CW'(W - 1 - i);
        end
    end
endmodule
