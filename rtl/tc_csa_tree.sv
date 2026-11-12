// Carry-save (Wallace-style) reduction of N two's-complement operands of W
// bits to two operands using 3:2 compressors. Recursive: each level turns
// every group of three operands into two and passes leftovers through.
// The final carry-propagate add is done by the instantiating module.
// All arithmetic is modulo 2^W, so W must hold the full signed sum.
module tc_csa_tree #(
    parameter int N = 9,
    parameter int W = 32
) (
    input  logic [N*W-1:0] in_flat,
    output logic [W-1:0]   sum_o,
    output logic [W-1:0]   carry_o
);
    if (N == 1) begin : g_one
        assign sum_o   = in_flat[W-1:0];
        assign carry_o = '0;
    end else if (N == 2) begin : g_two
        assign sum_o   = in_flat[W-1:0];
        assign carry_o = in_flat[2*W-1:W];
    end else begin : g_level
        localparam int G  = N / 3;          // 3:2 compressor groups at this level
        localparam int R  = N % 3;          // operands passed straight through
        localparam int NN = 2 * G + R;      // operands at the next level
        logic [NN*W-1:0] nxt;
        for (genvar g = 0; g < G; g++) begin : g_csa
            logic [W-1:0] x, y, z;
            assign x = in_flat[(3*g+0)*W +: W];
            assign y = in_flat[(3*g+1)*W +: W];
            assign z = in_flat[(3*g+2)*W +: W];
            // sum bits
            assign nxt[(2*g+0)*W +: W] = x ^ y ^ z;
            // majority bits weighted x2; the carry out of bit W-1 is dropped (mod 2^W)
            assign nxt[(2*g+1)*W +: W] = {((x[W-2:0] & y[W-2:0]) | (x[W-2:0] & z[W-2:0]) |
                                           (y[W-2:0] & z[W-2:0])), 1'b0};
        end
        for (genvar r = 0; r < R; r++) begin : g_pass
            assign nxt[(2*G+r)*W +: W] = in_flat[(3*G+r)*W +: W];
        end
        tc_csa_tree #(.N(NN), .W(W)) u_next (
            .in_flat (nxt),
            .sum_o   (sum_o),
            .carry_o (carry_o)
        );
    end
endmodule
