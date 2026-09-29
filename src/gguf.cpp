#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <type_traits>

// little-endian only; GGUF is also little-endian, so no byte swapping needed
namespace llm {

namespace {

constexpr size_t MAX_STRING_LEN = 1ull << 30;

// sequential reader over mmap buffer
struct binary_reader {
    const uint8_t* data;
    size_t         size;
    size_t         pos = 0;

    binary_reader(const uint8_t* data, size_t size) : data(data), size(size) {}

    void check(size_t n) {
        if (pos + n > size) LLM_ABORT("GGUF: unexpected end of file");
    }
    void bytes(void* dst, size_t n) {
        check(n);
        memcpy(dst, data + pos, n);
        pos += n;
    }
    template <class T> T read() {
        T v;
        bytes(&v, sizeof(T));
        return v;
    }
    std::string read_string() {
        uint64_t n = read<uint64_t>();
        if (n > MAX_STRING_LEN) LLM_ABORT("GGUF: string too long");
        check((size_t) n);
        std::string s((const char*) (data + pos), (size_t) n);
        pos += (size_t) n;
        return s;
    }
};


DType dtype_from_gguf(uint32_t t) {
    switch (t) {
        case 0:  return DType::F32;
        case 1:  return DType::F16;
        case 8:  return DType::Q8_0;
        case 12: return DType::Q4_K;
        case 13: return DType::Q5_K;
        case 14: return DType::Q6_K;
        default: LLM_ABORT("GGUF: unsupported tensor type %u", t);
    }
}


template <typename Dst, typename Src = Dst>
static void read_into(binary_reader& r, std::vector<Dst>& vec, uint64_t count) {
    vec.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<Dst, std::string>) {
            vec.emplace_back(r.read_string());
        } else {
            vec.emplace_back(static_cast<Dst>(r.read<Src>()));
        }
    }
}

void read_value(binary_reader& r, GGUFType type, uint64_t count, GGUFValue& val) {
    val.type = type;
    val.count = count;
    switch (type) {
        case GGUFType::UINT8:   read_into<int64_t, uint8_t> (r, val.ints, count); break;
        case GGUFType::INT8:    read_into<int64_t, int8_t>  (r, val.ints, count); break;
        case GGUFType::UINT16:  read_into<int64_t, uint16_t>(r, val.ints, count); break;
        case GGUFType::INT16:   read_into<int64_t, int16_t> (r, val.ints, count); break;
        case GGUFType::UINT32:  read_into<int64_t, uint32_t>(r, val.ints, count); break;
        case GGUFType::INT32:   read_into<int64_t, int32_t> (r, val.ints, count); break;
        case GGUFType::UINT64:  read_into<int64_t, uint64_t>(r, val.ints, count); break;
        case GGUFType::INT64:   read_into<int64_t, int64_t> (r, val.ints, count); break;
        case GGUFType::BOOL:    read_into<int64_t, uint8_t> (r, val.ints, count); break;
        case GGUFType::FLOAT32: read_into<double,  float>   (r, val.floats, count); break;
        case GGUFType::FLOAT64: read_into<double,  double>  (r, val.floats, count); break;
        case GGUFType::STRING:  read_into(r, val.str, count); break;
        default:
            LLM_ABORT("GGUF: invalid metadata type");
    }
}

size_t align_up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}

} // namespace

GGUF::~GGUF() {
    if (data) { munmap((void*) data, size); data = nullptr; }
    if (fd >= 0) { close(fd); fd = -1; }
}

const Tensor* GGUF::find_tensor(const std::string& name) const {
    auto it = tensor_index.find(name);
    if (it != tensor_index.end()) return &tensors[it->second];
    return nullptr;
}

const GGUFValue* GGUF::find_meta(const std::string& key) const {
    auto it = meta.find(key);
    return it == meta.end() ? nullptr : &it->second;
}

int64_t GGUF::get_int(const std::string& key, int64_t def) const {
    const GGUFValue* v = find_meta(key);
    return (v && !v->ints.empty()) ? v->ints[0] : def;
}

double GGUF::get_float(const std::string& key, double def) const {
    const GGUFValue* v = find_meta(key);
    return (v && !v->floats.empty()) ? v->floats[0] : def;
}

std::string GGUF::get_string(const std::string& key) const {
    const GGUFValue* v = find_meta(key);
    return (v && !v->str.empty()) ? v->str[0] : std::string{};
}

void load_gguf(GGUF& out, const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) LLM_ABORT("GGUF: cannot open %s", path.c_str());

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); LLM_ABORT("GGUF: fstat failed %s", path.c_str()); }
    size_t size = (size_t) st.st_size;
    if (size < 4) { close(fd); LLM_ABORT("GGUF: file too small %s", path.c_str()); }

    uint8_t* base = (uint8_t*) mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); LLM_ABORT("GGUF: mmap failed %s", path.c_str()); }

    out.fd = fd;
    out.size = size;
    out.data = base;

    binary_reader r(base, size);

    // ---- header ----
    char magic[4];
    r.bytes(magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0) LLM_ABORT("GGUF: bad magic %s", path.c_str());

    out.version = r.read<uint32_t>();
    if (out.version == 0 || out.version > 3) LLM_ABORT("GGUF: unsupported version %u", out.version);

    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv      = r.read<uint64_t>();

    // ---- metadata KV ----
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.read_string();
        GGUFType type = (GGUFType) r.read<uint32_t>();

        GGUFValue val;
        if (type == GGUFType::ARRAY) {
            GGUFType elem = (GGUFType) r.read<uint32_t>();
            uint64_t n = r.read<uint64_t>();
            val.is_array = true;
            read_value(r, elem, n, val);
        } else {
            read_value(r, type, 1, val);
        }
        out.meta[std::move(key)] = std::move(val);
    }

    // alignment (general.alignment, default 32)
    uint64_t alignment = out.get_int("general.alignment", 32);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        LLM_ABORT("GGUF: invalid alignment");
    }
    out.alignment = (size_t) alignment;

    // ---- tensor table ----
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = r.read_string();

        uint32_t n_dims = r.read<uint32_t>();
        if (n_dims < 1 || n_dims > 4) LLM_ABORT("GGUF: bad tensor rank %s", t.name.c_str());

        uint64_t ne[4] = {1, 1, 1, 1};
        for (uint32_t j = 0; j < n_dims; ++j) ne[j] = r.read<uint64_t>();

        uint32_t gtype = r.read<uint32_t>();
        t.type = dtype_from_gguf(gtype);

        t.offset = (size_t) r.read<uint64_t>();

        // flatten to 2D: ne0 = innermost, ne1 = product of remaining dims
        t.ne0 = ne[0];
        t.ne1 = ne[1] * ne[2] * ne[3];

        // row elements must be a multiple of block size
        type_traits ti = type_info(t.type);
        if (t.ne0 % ti.elems_per_block != 0) {
            LLM_ABORT("GGUF: tensor row not multiple of block size %s", t.name.c_str());
        }

        out.tensors.push_back(t);
    }

    // build name -> index map
    for (size_t i = 0; i < out.tensors.size(); ++i) {
        out.tensor_index[out.tensors[i].name] = i;
    }

    // data section starts at aligned offset; tensor offsets are relative to it
    size_t data_start = align_up(r.pos, out.alignment);
    for (auto& t : out.tensors) {
        t.data = base + data_start + t.offset;
    }
    out.data_offset = data_start;

    // metadata is parsed into C++ objects; release header pages
    if (data_start > 0) {
        madvise(base, data_start, MADV_DONTNEED);
    }
}

} // namespace llm
