// Fused dot-add lane:  d = c + sum_{i<K} a_i * b_i
//   a_i, b_i : FP16 or BF16 (in_fmt), c and d : FP32.
// Arithmetic (docs/01_numerics_spec.md, default "Ampere hypothesis"):
//   exact products -> align every term to the largest nominal exponent E,
//   keeping F fractional bits (bits shifted out are discarded, per term,
//   toward zero) -> exact fixed-point sum (carry-save tree + one adder) ->
//   single normalisation -> truncate to FP32 (RZ), overflow to infinity.
//
// Pipeline: NUM_STAGES (1..6) = latency in cycles. Register cuts are chosen
// from five candidate points; the output register is always present.
//   C1 after multiply   C2 after exponent max   C3 after alignment
//   C4 after CSA tree   C5 after final adder
module tc_fda #(
    parameter int K          = 8,    // products per lane
    parameter int F          = 24,   // alignment window: fractional bits below E
    parameter int NUM_STAGES = 3,    // 1..6
    parameter int SHARED_MUL = 1,
    parameter int ENABLE_BF16 = 1    // 0: FP16-only lane (area baseline; in_fmt ignored)
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    input  logic            in_fmt,        // 0: FP16, 1: BF16
    input  logic [K*16-1:0] a_i,           // a_i[16*i +: 16] = a[i]
    input  logic [K*16-1:0] b_i,
    input  logic [31:0]     c_i,
    output logic            out_valid,
    output logic [31:0]     d_o
);
    // ------------------------------------------------------------------
    // Widths
    // ------------------------------------------------------------------
    localparam int NT = K + 1;                       // terms: K products + c
    localparam int EW = 12;                          // signed exponent width
    localparam int TW = F + 2;                       // term magnitude (2 integer bits)
    localparam int SW = TW + 1 + $clog2(NT) + 1;     // signed sum width (with margin)
    localparam int MW = SW - 1;                      // |sum| width
    localparam int LW = $clog2(MW + 1);              // leading-zero count width

    // Register cut selection, bit i = cut C(i+1).
    localparam logic [4:0] CUTS =
        (NUM_STAGES <= 1) ? 5'b00000 :
        (NUM_STAGES == 2) ? 5'b00100 :
        (NUM_STAGES == 3) ? 5'b01001 :
        (NUM_STAGES == 4) ? 5'b10101 :
        (NUM_STAGES == 5) ? 5'b10111 : 5'b11111;

    // ------------------------------------------------------------------
    // Stage 0: unpack, multiply, special-value detection
    // ------------------------------------------------------------------
    logic [NT*TW-1:0] s0_mag;
    logic [NT*EW-1:0] s0_exp;
    logic [NT-1:0]    s0_neg, s0_zero;
    logic             s0_nan, s0_pinf, s0_ninf, s0_zneg;

    logic [K-1:0]     p_nan, p_pinf, p_ninf;
    logic             fmt;
    assign fmt = (ENABLE_BF16 != 0) ? in_fmt : 1'b0;

    for (genvar i = 0; i < K; i++) begin : g_prod
        logic          sa, sb, za, zb, ia, ib, na, nb;
        logic [10:0]   ma, mb;
        logic [EW-1:0] ea, eb;
        logic [21:0]   p;
        tc_unpack16 #(.EW(EW)) u_ua (.fmt(fmt), .x(a_i[16*i +: 16]), .sign(sa), .sig(ma),
                                    .exp(ea), .is_zero(za), .is_inf(ia), .is_nan(na));
        tc_unpack16 #(.EW(EW)) u_ub (.fmt(fmt), .x(b_i[16*i +: 16]), .sign(sb), .sig(mb),
                                    .exp(eb), .is_zero(zb), .is_inf(ib), .is_nan(nb));
        tc_mul #(.SHARED(SHARED_MUL)) u_mul (.fmt(fmt), .a(ma), .b(mb), .p(p));

        assign s0_mag [i*TW +: TW] = {p, {(F-20){1'b0}}};       // value = mag * 2^(e - F)
        assign s0_exp [i*EW +: EW] = ea + eb;                   // nominal exponent
        assign s0_neg [i]          = sa ^ sb;
        assign s0_zero[i]          = za | zb;
        assign p_nan  [i]          = na | nb | ((ia | ib) & (za | zb));
        assign p_pinf [i]          = (ia | ib) & ~(za | zb) & ~(na | nb) & ~(sa ^ sb);
        assign p_ninf [i]          = (ia | ib) & ~(za | zb) & ~(na | nb) &  (sa ^ sb);
    end

    if (1) begin : g_acc
        logic          sc, zc, ic, nc;
        logic [23:0]   mc;
        logic [EW-1:0] ec;
        tc_unpack32 #(.EW(EW)) u_uc (.x(c_i), .sign(sc), .sig(mc), .exp(ec),
                                    .is_zero(zc), .is_inf(ic), .is_nan(nc));
        assign s0_mag [K*TW +: TW] = {1'b0, mc, {(F-23){1'b0}}};
        assign s0_exp [K*EW +: EW] = ec;
        assign s0_neg [K]          = sc;
        assign s0_zero[K]          = zc;
        assign s0_nan              = (|p_nan) | nc;
        assign s0_pinf             = (|p_pinf) | (ic & ~sc);
        assign s0_ninf             = (|p_ninf) | (ic &  sc);
    end
    // -0 result only if every term is a zero carrying a negative sign
    assign s0_zneg = &(s0_zero & s0_neg);

    // ---- cut C1 ----
    logic             s1_v;
    logic [NT*TW-1:0] s1_mag;
    logic [NT*EW-1:0] s1_exp;
    logic [NT-1:0]    s1_neg, s1_zero;
    logic [3:0]       s1_flags;
    tc_pipe #(.W(1), .EN(CUTS[0]), .RST(1)) c1_v (.clk, .rst_n, .d(in_valid), .q(s1_v));
    tc_pipe #(.W(NT*TW + NT*EW + 2*NT + 4), .EN(CUTS[0])) c1_d (.clk, .rst_n,
        .d({s0_mag, s0_exp, s0_neg, s0_zero, s0_nan, s0_pinf, s0_ninf, s0_zneg}),
        .q({s1_mag, s1_exp, s1_neg, s1_zero, s1_flags}));

    // ------------------------------------------------------------------
    // Stage 1: reference exponent E = max nominal exponent of non-zero terms
    // (balanced comparator tree; zero terms do not participate)
    // ------------------------------------------------------------------
    // Exponents are two's complement; comparing them with the sign bit inverted
    // (offset binary) turns a signed compare into an unsigned one.
    localparam int NP = 1 << $clog2(NT);
    logic [EW-1:0] s1_emax;
    logic [NP*EW-1:0] ev;
    logic [NP-1:0]    vv;
    always_comb begin
        for (int i = 0; i < NP; i++) begin
            if (i < NT) begin
                ev[i*EW +: EW] = s1_exp[i*EW +: EW];
                vv[i]          = ~s1_zero[i];
            end else begin
                ev[i*EW +: EW] = '0;
                vv[i]          = 1'b0;
            end
        end
        for (int span = NP / 2; span >= 1; span = span / 2) begin
            for (int i = 0; i < span; i++) begin
                logic [EW-1:0] lo, hi;
                logic          take_hi;
                lo      = ev[(2*i)*EW +: EW];
                hi      = ev[(2*i+1)*EW +: EW];
                take_hi = vv[2*i+1] &&
                          (!vv[2*i] || ({~hi[EW-1], hi[EW-2:0]} > {~lo[EW-1], lo[EW-2:0]}));
                ev[i*EW +: EW] = take_hi ? hi : lo;
                vv[i]          = vv[2*i] | vv[2*i+1];
            end
        end
        s1_emax = ev[EW-1:0];
    end

    // ---- cut C2 ----
    logic             s2_v;
    logic [NT*TW-1:0] s2_mag;
    logic [NT*EW-1:0] s2_exp;
    logic [NT-1:0]    s2_neg, s2_zero;
    logic [3:0]       s2_flags;
    logic [EW-1:0]    s2_emax;
    tc_pipe #(.W(1), .EN(CUTS[1]), .RST(1)) c2_v (.clk, .rst_n, .d(s1_v), .q(s2_v));
    tc_pipe #(.W(NT*TW + NT*EW + 2*NT + 4 + EW), .EN(CUTS[1])) c2_d (.clk, .rst_n,
        .d({s1_mag, s1_exp, s1_neg, s1_zero, s1_flags, s1_emax}),
        .q({s2_mag, s2_exp, s2_neg, s2_zero, s2_flags, s2_emax}));

    // ------------------------------------------------------------------
    // Stage 2: align to the window [E - F, ...]; discard shifted-out bits
    // (truncate magnitude, then apply sign) and sign-extend to SW bits.
    // ------------------------------------------------------------------
    logic [NT*SW-1:0] s2_al;
    always_comb begin
        for (int i = 0; i < NT; i++) begin
            logic [EW-1:0] sh;
            logic [TW-1:0] m;
            logic [SW-1:0] v;
            sh = s2_emax - s2_exp[i*EW +: EW];   // >= 0 for every non-zero term
            if (s2_zero[i] || (sh >= EW'(TW))) m = '0;
            else                               m = s2_mag[i*TW +: TW] >> sh;
            v = SW'(m);
            s2_al[i*SW +: SW] = s2_neg[i] ? (~v + 1'b1) : v;
        end
    end

    // ---- cut C3 ----
    logic             s3_v;
    logic [NT*SW-1:0] s3_al;
    logic [3:0]       s3_flags;
    logic [EW-1:0]    s3_emax;
    tc_pipe #(.W(1), .EN(CUTS[2]), .RST(1)) c3_v (.clk, .rst_n, .d(s2_v), .q(s3_v));
    tc_pipe #(.W(NT*SW + 4 + EW), .EN(CUTS[2])) c3_d (.clk, .rst_n,
        .d({s2_al, s2_flags, s2_emax}), .q({s3_al, s3_flags, s3_emax}));

    // ------------------------------------------------------------------
    // Stage 3: carry-save reduction of the NT aligned terms
    // ------------------------------------------------------------------
    logic [SW-1:0] s3_cs_s, s3_cs_c;
    tc_csa_tree #(.N(NT), .W(SW)) u_csa (.in_flat(s3_al), .sum_o(s3_cs_s), .carry_o(s3_cs_c));

    // ---- cut C4 ----
    logic          s4_v;
    logic [SW-1:0] s4_cs_s, s4_cs_c;
    logic [3:0]    s4_flags;
    logic [EW-1:0] s4_emax;
    tc_pipe #(.W(1), .EN(CUTS[3]), .RST(1)) c4_v (.clk, .rst_n, .d(s3_v), .q(s4_v));
    tc_pipe #(.W(2*SW + 4 + EW), .EN(CUTS[3])) c4_d (.clk, .rst_n,
        .d({s3_cs_s, s3_cs_c, s3_flags, s3_emax}), .q({s4_cs_s, s4_cs_c, s4_flags, s4_emax}));

    // ------------------------------------------------------------------
    // Stage 4: final carry-propagate adder
    // ------------------------------------------------------------------
    logic [SW-1:0] s4_sum;
    assign s4_sum = s4_cs_s + s4_cs_c;

    // ---- cut C5 ----
    logic          s5_v;
    logic [SW-1:0] s5_sum;
    logic [3:0]    s5_flags;
    logic [EW-1:0] s5_emax;
    tc_pipe #(.W(1), .EN(CUTS[4]), .RST(1)) c5_v (.clk, .rst_n, .d(s4_v), .q(s5_v));
    tc_pipe #(.W(SW + 4 + EW), .EN(CUTS[4])) c5_d (.clk, .rst_n,
        .d({s4_sum, s4_flags, s4_emax}), .q({s5_sum, s5_flags, s5_emax}));

    // ------------------------------------------------------------------
    // Stage 5: normalise once, truncate to FP32, apply special values
    // ------------------------------------------------------------------
    logic          s5_neg;
    /* verilator lint_off UNUSEDSIGNAL */      // low bits of s5_norm are truncated (RZ)
    logic [MW-1:0] s5_mag, s5_norm;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [LW-1:0] s5_lz;
    logic [31:0]   s5_d;

    assign s5_neg = s5_sum[SW-1];
    always_comb begin
        /* verilator lint_off UNUSEDSIGNAL */  // t[SW-1] is the sign, always 0 after abs
        logic [SW-1:0] t;
        /* verilator lint_on UNUSEDSIGNAL */
        t      = s5_neg ? (~s5_sum + 1'b1) : s5_sum;
        s5_mag = t[MW-1:0];                 // |sum| < 2^(SW-1) by construction
    end
    tc_lzc #(.W(MW)) u_lzc (.x(s5_mag), .cnt(s5_lz));

    always_comb begin
        logic          f_nan, f_pinf, f_ninf, f_zneg;
        logic [EW+1:0] be;                  // biased exponent of the leading one (two's complement)
        logic [EW+1:0] rsh;
        logic [23:0]   sig24;
        /* verilator lint_off UNUSEDSIGNAL */  // sub[23] is always 0 (shift >= 1)
        logic [23:0]   sub;
        /* verilator lint_on UNUSEDSIGNAL */
        {f_nan, f_pinf, f_ninf, f_zneg} = s5_flags;
        sub     = '0;

        s5_norm = s5_mag << s5_lz;          // leading one at bit MW-1
        sig24   = s5_norm[MW-1 -: 24];
        // leading-one index p = MW-1-lz;  exponent = E - F + p;  biased = exponent + 127
        be      = {{2{s5_emax[EW-1]}}, s5_emax} + (EW+2)'(127 + MW - 1 - F)
                - {{(EW+2-LW){1'b0}}, s5_lz};
        rsh     = (EW+2)'(1) - be;          // subnormal shift, used only when be <= 0

        if (s5_mag == '0) begin
            s5_d = {f_zneg, 31'd0};
        end else if (!be[EW+1] && (be > (EW+2)'(254))) begin
            s5_d = {s5_neg, 8'hFF, 23'd0};                       // overflow -> inf
        end else if (!be[EW+1] && (be != '0)) begin
            s5_d = {s5_neg, be[7:0], sig24[22:0]};               // normal, truncated
        end else begin
            sub  = (rsh >= (EW+2)'(24)) ? 24'd0 : (sig24 >> rsh);
            s5_d = {s5_neg, 8'h00, sub[22:0]};                   // subnormal, truncated
        end

        if (f_nan || (f_pinf && f_ninf)) s5_d = 32'h7FFF_FFFF;
        else if (f_pinf)                 s5_d = 32'h7F80_0000;
        else if (f_ninf)                 s5_d = 32'hFF80_0000;
    end

    // ---- output register (always present) ----
    tc_pipe #(.W(1), .EN(1'b1), .RST(1)) co_v (.clk, .rst_n, .d(s5_v), .q(out_valid));
    tc_pipe #(.W(32), .EN(1'b1)) co_d (.clk, .rst_n, .d(s5_d), .q(d_o));

    // Elaboration-time parameter checks
`ifndef SYNTHESIS
    initial begin
        if (F < 23) $fatal(1, "tc_fda: F must be >= 23 (FP32 accumulator needs 23 fractional bits)");
        if (NUM_STAGES < 1 || NUM_STAGES > 6) $fatal(1, "tc_fda: NUM_STAGES must be 1..6");
    end
`endif
endmodule
