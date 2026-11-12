// Optional pipeline register: a flop when EN=1, a wire when EN=0.
// Data registers have no reset; only the valid bit (RST=1) is reset.
module tc_pipe #(
    parameter int W   = 1,
    parameter bit EN  = 1'b1,
    parameter bit RST = 1'b0
) (
    /* verilator lint_off UNUSEDSIGNAL */  // unused when EN=0 (wire) or RST=0
    input  logic         clk,
    input  logic         rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [W-1:0] d,
    output logic [W-1:0] q
);
    if (EN) begin : g_reg
        if (RST) begin : g_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) q <= '0;
                else        q <= d;
            end
        end else begin : g_norst
            always_ff @(posedge clk) q <= d;
        end
    end else begin : g_wire
        assign q = d;
    end
endmodule
