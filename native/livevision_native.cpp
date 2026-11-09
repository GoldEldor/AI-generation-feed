// native/livevision_native.cpp
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

inline uint32_t be16(const uint8_t* p) {
    return (uint32_t(p[0]) << 8) | uint32_t(p[1]);
}

inline bool type_is(const uint8_t* p, const char* t) {
    return std::memcmp(p, t, 4) == 0;
}

bool copy_out(const uint8_t* in, size_t from, size_t len,
              uint8_t* out, size_t cap, size_t& o) {
    if (len > cap - o) return false;
    std::memcpy(out + o, in + from, len);
    o += len;
    return true;
}

long long strip_jpeg(const uint8_t* in, size_t n, uint8_t* out, size_t cap) {
    size_t o = 0;
    if (!copy_out(in, 0, 2, out, cap, o)) return -1;
    size_t i = 2;
    while (i < n) {
        if (in[i] != 0xFF) return -1;
        size_t j = i;
        while (j < n && in[j] == 0xFF) ++j;
        if (j >= n) return -1;
        const uint8_t m = in[j];
        if (m == 0x00) return -1;

        if (m == 0xD9) {
            return copy_out(in, i, j + 1 - i, out, cap, o) ? (long long)o : -1;
        }
        if (m == 0x01 || (m >= 0xD0 && m <= 0xD7)) {
            if (!copy_out(in, i, j + 1 - i, out, cap, o)) return -1;
            i = j + 1;
            continue;
        }

        if (j + 3 > n) return -1;
        const size_t seg_end = j + 1 + be16(in + j + 1);
        if (seg_end > n) return -1;

        if (m == 0xDA) {
            if (!copy_out(in, i, n - i, out, cap, o)) return -1;
            return (long long)o;
        }

        const bool drop = (m >= 0xE1 && m <= 0xEF && m != 0xEE) || m == 0xFE;
        if (!drop && !copy_out(in, i, seg_end - i, out, cap, o)) return -1;
        i = seg_end;
    }
    return -1;
}

long long strip_png(const uint8_t* in, size_t n, uint8_t* out, size_t cap) {
    if (n < 8) return -1;
    size_t o = 0;
    if (!copy_out(in, 0, 8, out, cap, o)) return -1;
    size_t i = 8;
    while (i + 12 <= n) {
        const size_t len = be32(in + i);
        if (len > n - i - 12) return -1;
        const size_t total = 12 + len;
        const uint8_t* type = in + i + 4;
        const bool drop = type_is(type, "tEXt") || type_is(type, "zTXt") ||
                          type_is(type, "iTXt") || type_is(type, "eXIf");
        if (!drop && !copy_out(in, i, total, out, cap, o)) return -1;
        i += total;
        if (type_is(type, "IEND")) return (long long)o;
    }
    return -1;
}

}  // namespace

extern "C" long long lv_strip_metadata(const uint8_t* in, size_t n,
                                       uint8_t* out, size_t cap) {
    if (!in || !out || n == 0 || cap < n) return -1;
    if (n >= 2 && in[0] == 0xFF && in[1] == 0xD8) return strip_jpeg(in, n, out, cap);
    if (n >= 8 && std::memcmp(in, "\x89PNG\r\n\x1a\n", 8) == 0) return strip_png(in, n, out, cap);
    return -1;
}
