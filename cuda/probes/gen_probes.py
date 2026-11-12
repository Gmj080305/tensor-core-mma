#!/usr/bin/env python3
"""Generate tensor-core probe tiles (16x16x16) for tc_probe and the model.

Each *targeted* family isolates one numerical feature from
docs/01_numerics_spec.md. A test occupies one diagonal element of a tile:
row d of A, column d of B and C[d][d], so one tile carries up to 16 tests.
Every output element of every tile is still compared, so the off-diagonal
results are extra (cross) tests.

*Random* families fill whole tiles.

Outputs (per format):  <out>/<fmt>.bin  and  <out>/<fmt>.families.txt

    python3 gen_probes.py --out ../../probes --seed 1 --random-tiles 200
"""
import argparse
import math
import os
import random
import struct

T = 16


# --------------------------------------------------------------- encoding --
def enc16(fmt, x):
    """Exact (round-to-nearest-even) encoding of x in FP16 or BF16."""
    if isinstance(x, str):  # raw specials
        return SPECIALS[fmt][x]
    if fmt == "fp16":
        return struct.unpack("<H", struct.pack("<e", x))[0]
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def enc32(x):
    if isinstance(x, str):
        return {"+inf": 0x7F800000, "-inf": 0xFF800000, "nan": 0x7FC00000, "-0": 0x80000000}[x]
    return struct.unpack("<I", struct.pack("<f", x))[0]


SPECIALS = {
    "fp16": {"+inf": 0x7C00, "-inf": 0xFC00, "nan": 0x7E00, "-0": 0x8000},
    "bf16": {"+inf": 0x7F80, "-inf": 0xFF80, "nan": 0x7FC0, "-0": 0x8000},
}


def p2(n):
    return math.ldexp(1.0, n)


def split_pow2(n, fmt):
    """Two in-range powers of two whose product is 2^n."""
    lo = -14 if fmt == "fp16" else -126
    a = n // 2
    b = n - a
    if fmt == "fp16":
        a = max(a, -24)
        b = n - a
        if b < -24:
            raise ValueError("2^%d not reachable with fp16" % n)
    return p2(a), p2(b)


# ------------------------------------------------------------------ tiles --
class Builder:
    def __init__(self, fmt):
        self.fmt = fmt
        self.tiles = []        # list of (A, B, C)
        self.families = []     # (name, first, count)
        self._cur = None
        self._slot = T

    def begin(self, name):
        self._flush()
        self._fam_name = name
        self._fam_first = len(self.tiles)
        self._slot = T

    def end(self):
        self._flush()
        self.families.append((self._fam_name, self._fam_first, len(self.tiles) - self._fam_first))

    def _flush(self):
        if self._cur is not None:
            self.tiles.append(self._cur)
            self._cur = None
        self._slot = T

    def test(self, pairs, c=0.0):
        """pairs: list of (k, a, b) with a, b floats or special names."""
        if self._slot >= T:
            self._flush()
            self._cur = ([0] * (T * T), [0] * (T * T), [0] * (T * T))
            self._slot = 0
        d = self._slot
        A, B, C = self._cur
        for k, a, b in pairs:
            A[d * T + k] = enc16(self.fmt, a)
            B[k * T + d] = enc16(self.fmt, b)
        C[d * T + d] = enc32(c)
        self._slot += 1

    def raw_tile(self, A, B, C):
        self._flush()
        self.tiles.append((A, B, C))


def targeted(bld, fmt):
    # A zero times `big` has nominal exponent emin + e(big) = +1 in both formats,
    # i.e. above the exponent (0) of the other terms in the zero_in_emax tests.
    big = 65504.0 if fmt == "fp16" else p2(127)

    bld.begin("align_bits")  # 1 - 1 + 2^-n: survives iff n <= F
    for n in range(18, 35):
        a, b = split_pow2(-n, fmt)
        bld.test([(0, 1.0, 1.0), (1, 1.0, -1.0), (2, a, b)])
        bld.test([(0, -1.0, 1.0), (1, a, b)], c=1.0)          # c is the big term
        bld.test([(8, 1.0, 1.0), (9, 1.0, -1.0), (10, a, b)])  # second block
    bld.end()

    bld.begin("block_size")  # [1,-1] at (p,p+1), 2^-30 at j: nonzero iff different block
    a30, b30 = split_pow2(-30, fmt)
    for p in (0, 4, 8, 12):
        for j in range(T):
            if j in (p, p + 1):
                continue
            bld.test([(p, 1.0, 1.0), (p + 1, 1.0, -1.0), (j, a30, b30)])
    bld.end()

    bld.begin("chain_order")  # does c enter the first block?
    for n in (26, 28, 30, 34):
        bld.test([(0, 1.0, 1.0), (1, 1.0, -1.0)], c=p2(-n))
        bld.test([(8, 1.0, 1.0), (9, 1.0, -1.0)], c=p2(-n))
        bld.test([(0, 1.0, 1.0), (9, 1.0, -1.0)], c=p2(-n))
    bld.end()

    bld.begin("trunc_mode")  # 1 + (-2^-n) beyond the window: 1.0 (signmag) vs 1-2^-24 (floor)
    for n in range(25, 41):
        a, b = split_pow2(-n, fmt)
        bld.test([(0, 1.0, 1.0), (1, -a, b)])
        bld.test([(0, -1.0, 1.0), (1, a, b)])
        bld.test([(0, 1.0, 1.0)], c=-p2(-n))
    bld.end()

    bld.begin("final_round")  # remainders below / at / above half an ulp, both signs
    u23 = split_pow2(-23, fmt)
    for s in (1.0, -1.0):
        for k in (1, 2, 3):
            t = split_pow2(-24, fmt)
            bld.test([(0, s, 1.0), (1, s * u23[0], u23[1]), (2, s * k * t[0], t[1])])
            bld.test([(0, s, 1.0), (1, s * k * t[0], t[1])])
            bld.test([(0, s * 1.5, 1.0), (1, s * k * t[0], t[1])])
    bld.end()

    bld.begin("zero_in_emax")  # a zero product with a large nominal exponent
    # 1 - 1 + 2^-n: with E=0 the small term survives for n <= F; if the zero
    # product lifts E to 1 it survives only for n <= F-1. Sweeping n covers any F.
    for n in range(22, 31):
        tn = split_pow2(-n, fmt)
        for z in (0.0, "-0"):
            bld.test([(0, z, big), (1, 1.0, 1.0), (2, 1.0, -1.0), (3, tn[0], tn[1])])
    bld.end()

    bld.begin("align_ref_product_msb")  # 2.25 has its leading one at 2^1, nominal exponent 0
    for n in range(22, 31):
        tn = split_pow2(-n, fmt)
        for x in (1.5, 1.75):
            bld.test([(0, x, x), (1, x, -x), (2, tn[0], tn[1])])
            bld.test([(0, x, x), (1, -x, x), (2, -tn[0], tn[1])])
    bld.end()

    bld.begin("align_ref_subnormal")
    if fmt == "fp16":
        for sub in (p2(-24), p2(-20), 3 * p2(-24)):
            bld.test([(0, sub, 16384.0), (1, p2(-15), p2(-15))])
            bld.test([(0, sub, 16384.0), (1, -p2(-15), p2(-15))])
    else:
        for sub in (p2(-133), p2(-130), 3 * p2(-133)):
            bld.test([(0, sub, p2(126)), (1, p2(-20), p2(-30))])
            bld.test([(0, sub, p2(126)), (1, -p2(-20), p2(-30))])
    bld.end()

    bld.begin("subnormal_in")  # exact results from subnormal inputs
    subs = [p2(-24), 5 * p2(-24), 1023 * p2(-24)] if fmt == "fp16" else [p2(-133), 5 * p2(-133), 127 * p2(-133)]
    for s in subs:
        for m in (1.0, -3.0, 1024.0):
            bld.test([(0, s, m)])
            bld.test([(0, s, m), (5, s, m), (11, -s, m)])
    bld.end()

    bld.begin("subnormal_out")
    if fmt == "bf16":
        for n in (-130, -135, -140, -145, -149, -150, -152):
            a, b = split_pow2(n, fmt)
            bld.test([(0, a, b)])
            bld.test([(0, -a, b)])
            bld.test([(0, a, b), (1, a, b), (9, a, 3 * b)])
            bld.test([(0, a, b)], c=p2(-126))
    for n in (-127, -130, -140, -149):
        bld.test([(0, 0.0, 1.0)], c=p2(n))
        bld.test([(0, 1.0, 1.0), (1, -1.0, 1.0)], c=-p2(n))
    bld.end()

    bld.begin("overflow")
    if fmt == "bf16":
        for x in (1.0, 1.5, 1.9921875):
            bld.test([(0, p2(127), x), (1, p2(127), x)])
            bld.test([(0, -p2(127), x), (1, -p2(127), x)])
            bld.test([(0, p2(127), x)], c=3.0e38)
            bld.test([(0, p2(127), x), (8, p2(127), x)])
    bld.test([(0, 65504.0 if fmt == "fp16" else p2(100), 65504.0 if fmt == "fp16" else p2(27))], c=3.4e38)
    bld.test([(0, 1.0, 1.0)], c=3.4028234663852886e38)
    bld.end()

    bld.begin("specials")
    bld.test([(0, "+inf", 0.0)])
    bld.test([(0, "+inf", 1.0), (1, "-inf", 1.0)])
    bld.test([(0, "+inf", 1.0)], c="-inf")
    bld.test([(0, "nan", 1.0)])
    bld.test([(9, "nan", 0.0)])
    bld.test([(0, "+inf", -2.0)], c=5.0)
    bld.test([(12, "-inf", "-inf")])
    bld.test([(k, "-0", 1.0) for k in range(T)], c="-0")
    bld.test([(k, "-0", 1.0) for k in range(8)], c="-0")
    bld.test([(k, "-0", 1.0) for k in range(T)], c=0.0)
    bld.test([(0, 3.0, 5.0), (1, 3.0, -5.0)], c="-0")
    bld.test([(0, 1.0, 1.0)], c="nan")
    bld.end()


def random_tiles(bld, fmt, rng, n_tiles):
    def rnd(lo, hi, p_sub=0.0):
        s = -1.0 if rng.random() < 0.5 else 1.0
        if rng.random() < p_sub:
            m = rng.randint(1, 1023 if fmt == "fp16" else 127)
            return s * m * p2(-24 if fmt == "fp16" else -133)
        return s * (1.0 + rng.random()) * p2(rng.randint(lo, hi))

    emin, emax = (-14, 15) if fmt == "fp16" else (-126, 127)
    for name, gen in (
        ("random_normal", lambda: (rnd(-3, 3), rnd(-3, 3))),
        ("random_wide", lambda: (rnd(emin, emax), rnd(emin, emax))),
        ("random_subnormal", lambda: (rnd(emin, emin + 4, 0.5), rnd(-4, 6, 0.2))),
    ):
        bld.begin(name)
        for _ in range(n_tiles):
            A = [enc16(fmt, gen()[0]) for _ in range(T * T)]
            B = [enc16(fmt, gen()[1]) for _ in range(T * T)]
            C = [enc32(rnd(-6, 6)) if rng.random() < 0.8 else 0 for _ in range(T * T)]
            bld.raw_tile(A, B, C)
        bld.end()

    bld.begin("random_cancel")  # A[i][k+1] = -A[i][k], B rows duplicated: products cancel
    for _ in range(n_tiles):
        A, B = [0] * (T * T), [0] * (T * T)
        for i in range(T):
            for k in range(0, T, 2):
                v = rnd(-2, 2)
                A[i * T + k] = enc16(fmt, v)
                A[i * T + k + 1] = enc16(fmt, -v)
        for k in range(0, T, 2):
            for j in range(T):
                v = rnd(-2, 2)
                B[k * T + j] = enc16(fmt, v)
                B[(k + 1) * T + j] = enc16(fmt, v)
        for _ in range(3):  # perturb a few entries so small residues remain
            A[rng.randrange(T * T)] = enc16(fmt, rnd(-14, -8))
        C = [enc32(rnd(-30, -20)) if rng.random() < 0.5 else 0 for _ in range(T * T)]
        bld.raw_tile(A, B, C)
    bld.end()


def write(bld, fmt, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fmt + ".bin")
    with open(path, "wb") as f:
        f.write(b"TCMP")
        f.write(struct.pack("<6I", 1, 0 if fmt == "fp16" else 1, len(bld.tiles), T, T, T))
        for A, B, C in bld.tiles:
            f.write(struct.pack("<%dH" % (T * T), *A))
            f.write(struct.pack("<%dH" % (T * T), *B))
            f.write(struct.pack("<%dI" % (T * T), *C))
    fam_path = os.path.join(out_dir, fmt + ".families.txt")
    with open(fam_path, "w") as f:
        f.write("# family first_tile num_tiles\n")
        for name, first, num in bld.families:
            f.write("%s %d %d\n" % (name, first, num))
    return path, fam_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="probes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random-tiles", type=int, default=200, help="tiles per random family")
    ap.add_argument("--formats", default="fp16,bf16")
    args = ap.parse_args()
    for fmt in args.formats.split(","):
        rng = random.Random(args.seed)
        bld = Builder(fmt)
        targeted(bld, fmt)
        random_tiles(bld, fmt, rng, args.random_tiles)
        path, fam = write(bld, fmt, args.out)
        print("%s: %d tiles in %d families -> %s, %s" % (fmt, len(bld.tiles), len(bld.families), path, fam))


if __name__ == "__main__":
    main()
