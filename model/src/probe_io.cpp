#include "tcmma/probe_io.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tcmma {

namespace {

bool rd32(std::FILE* f, uint32_t& v) { return std::fread(&v, 4, 1, f) == 1; }
bool wr32(std::FILE* f, uint32_t v) { return std::fwrite(&v, 4, 1, f) == 1; }

template <typename T>
bool rdvec(std::FILE* f, T* dst, size_t n) {
    return std::fread(dst, sizeof(T), n, f) == n;
}

template <typename T>
bool wrvec(std::FILE* f, const T* src, size_t n) {
    return std::fwrite(src, sizeof(T), n, f) == n;
}

struct FileCloser {
    std::FILE* f;
    ~FileCloser() {
        if (f) std::fclose(f);
    }
};

}  // namespace

bool read_probes(const std::string& path, ProbeSet& out, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    FileCloser fc{f};
    char magic[4];
    uint32_t ver, fmt, count;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "TCMP", 4) != 0) {
        err = path + ": bad magic (expected TCMP)";
        return false;
    }
    if (!rd32(f, ver) || ver != 1 || !rd32(f, fmt) || !rd32(f, count) || !rd32(f, out.M) ||
        !rd32(f, out.N) || !rd32(f, out.K)) {
        err = path + ": bad header";
        return false;
    }
    if (fmt > 1) { err = path + ": unknown format id"; return false; }
    out.fmt = static_cast<InFmt>(fmt);
    const size_t mk = size_t(out.M) * out.K, kn = size_t(out.K) * out.N, mn = size_t(out.M) * out.N;
    out.A.resize(count * mk);
    out.B.resize(count * kn);
    out.C.resize(count * mn);
    for (uint32_t i = 0; i < count; ++i) {
        if (!rdvec(f, &out.A[i * mk], mk) || !rdvec(f, &out.B[i * kn], kn) ||
            !rdvec(f, &out.C[i * mn], mn)) {
            err = path + ": truncated at case " + std::to_string(i);
            return false;
        }
    }
    return true;
}

bool write_probes(const std::string& path, const ProbeSet& ps, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot create " + path; return false; }
    FileCloser fc{f};
    const uint32_t count = ps.count();
    std::fwrite("TCMP", 1, 4, f);
    wr32(f, 1);
    wr32(f, static_cast<uint32_t>(ps.fmt));
    wr32(f, count);
    wr32(f, ps.M);
    wr32(f, ps.N);
    wr32(f, ps.K);
    const size_t mk = size_t(ps.M) * ps.K, kn = size_t(ps.K) * ps.N, mn = size_t(ps.M) * ps.N;
    for (uint32_t i = 0; i < count; ++i) {
        if (!wrvec(f, &ps.A[i * mk], mk) || !wrvec(f, &ps.B[i * kn], kn) ||
            !wrvec(f, &ps.C[i * mn], mn)) {
            err = "write failed: " + path;
            return false;
        }
    }
    return true;
}

bool read_results(const std::string& path, ResultSet& out, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    FileCloser fc{f};
    char magic[4];
    uint32_t ver, count;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "TCMD", 4) != 0) {
        err = path + ": bad magic (expected TCMD)";
        return false;
    }
    if (!rd32(f, ver) || ver != 1 || !rd32(f, count) || !rd32(f, out.M) || !rd32(f, out.N)) {
        err = path + ": bad header";
        return false;
    }
    const size_t mn = size_t(out.M) * out.N;
    out.D.resize(count * mn);
    if (!rdvec(f, out.D.data(), out.D.size())) { err = path + ": truncated"; return false; }
    return true;
}

bool write_results(const std::string& path, const ResultSet& rs, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot create " + path; return false; }
    FileCloser fc{f};
    std::fwrite("TCMD", 1, 4, f);
    wr32(f, 1);
    wr32(f, rs.count());
    wr32(f, rs.M);
    wr32(f, rs.N);
    if (!wrvec(f, rs.D.data(), rs.D.size())) { err = "write failed: " + path; return false; }
    return true;
}

bool read_families(const std::string& path, std::vector<Family>& out, std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        Family fam;
        if (!(ls >> fam.name >> fam.first >> fam.num)) {
            err = path + ": bad line: " + line;
            return false;
        }
        out.push_back(fam);
    }
    return true;
}

}  // namespace tcmma
