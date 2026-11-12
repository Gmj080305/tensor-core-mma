#!/usr/bin/env python3
"""Compare a GPU result dump with a model result dump, family by family.

    python3 scripts/compare_dumps.py --probes probes/fp16.bin \
        --gpu results/fp16.gpu.out --model results/fp16.model.out \
        --families probes/fp16.families.txt [--strict-nan] [--show 5]

Exit status is 0 when every element matches, 1 otherwise.
"""
import argparse
import struct
import sys

T_DEFAULT = 16


def read_probes(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"TCMP":
        sys.exit("%s: not a probe file" % path)
    ver, fmt, count, M, N, K = struct.unpack_from("<6I", data, 4)
    off = 28
    tiles = []
    for _ in range(count):
        A = struct.unpack_from("<%dH" % (M * K), data, off); off += 2 * M * K
        B = struct.unpack_from("<%dH" % (K * N), data, off); off += 2 * K * N
        C = struct.unpack_from("<%dI" % (M * N), data, off); off += 4 * M * N
        tiles.append((A, B, C))
    return ("fp16" if fmt == 0 else "bf16"), (M, N, K), tiles


def read_results(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"TCMD":
        sys.exit("%s: not a result file" % path)
    ver, count, M, N = struct.unpack_from("<4I", data, 4)
    vals = struct.unpack_from("<%dI" % (count * M * N), data, 20)
    return (M, N), [vals[i * M * N:(i + 1) * M * N] for i in range(count)]


def read_families(path, n_tiles):
    fams = []
    if path:
        with open(path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    name, first, num = line.split()
                    fams.append((name, int(first), int(num)))
    return fams or [("all", 0, n_tiles)]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def is_nan(b):
    return (b >> 23) & 0xFF == 0xFF and b & 0x7FFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--families")
    ap.add_argument("--strict-nan", action="store_true")
    ap.add_argument("--show", type=int, default=5, help="mismatches to print per family")
    a = ap.parse_args()

    fmt, (M, N, K), tiles = read_probes(a.probes)
    _, gpu = read_results(a.gpu)
    _, mod = read_results(a.model)
    if not (len(tiles) == len(gpu) == len(mod)):
        sys.exit("tile counts differ: probes %d gpu %d model %d" % (len(tiles), len(gpu), len(mod)))
    fams = read_families(a.families, len(tiles))

    total_bad = 0
    print("%-26s %10s %10s" % ("family (" + fmt + ")", "elements", "mismatch"))
    for name, first, num in fams:
        bad, shown = 0, 0
        for t in range(first, first + num):
            for e in range(M * N):
                g, m = gpu[t][e], mod[t][e]
                if g == m or (not a.strict_nan and is_nan(g) and is_nan(m)):
                    continue
                bad += 1
                if shown < a.show:
                    i, j = divmod(e, N)
                    A, B, C = tiles[t]
                    nz = [k for k in range(K) if A[i * K + k] or B[k * N + j]]
                    print("    tile %d D[%d][%d]: gpu %08X (%.9g)  model %08X (%.9g)  c=%08X nonzero k=%s"
                          % (t, i, j, g, f32(g), m, f32(m), C[e], nz))
                    shown += 1
        total_bad += bad
        print("%-26s %10d %10d" % (name, num * M * N, bad))
    print("TOTAL mismatches: %d" % total_bad)
    sys.exit(0 if total_bad == 0 else 1)


if __name__ == "__main__":
    main()
