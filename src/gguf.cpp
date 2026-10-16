
#include "gguf.h"

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <type_traits>

namespace llm {

namespace {

class binary_reader {
public:
    binary_reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    void read_bytes(void* dst, size_t n) {
        check(n);
        memcpy(dst, data_ + pos_, n);
        pos_ += n;
    }
    template <class T> T read() {
        T v;
        read_bytes(&v, sizeof(T));
        return v;
    }
    std::string read_string() {
        uint64_t n = read<uint64_t>();
        if (n > (1ull << 30)) LLM_ABORT("GGUF: string too long");
        check(n);
        std::string s((const char*)(data_ + pos_), n);
        pos_ += n;
        return s;
    }
    size_t pos() const { return pos_; }

private:
    void check(size_t n) {
        if (pos_ + n > size_) LLM_ABORT("GGUF: unexpected end of file");
    }
    const uint8_t* data_;
    size_t         size_;
    size_t         pos_ = 0;
};

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
        default: LLM_ABORT("GGUF: invalid metadata type");
    }
}

} // namespace

GGUF::~GGUF() {
    if (data_) munmap((void*)data_, size_);
    if (fd_ >= 0) close(fd_);
}

void GGUF::load(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) LLM_ABORT("GGUF: cannot open %s", path.c_str());

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        LLM_ABORT("GGUF: fstat failed %s", path.c_str());
    }
    size_t size = (size_t)st.st_size;
    if (size < 4) {
        close(fd);
        LLM_ABORT("GGUF: file too small %s", path.c_str());
    }

    uint8_t* base = (uint8_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        LLM_ABORT("GGUF: mmap failed %s", path.c_str());
    }

    fd_ = fd;
    size_ = size;
    data_ = base;

    binary_reader r(base, size);

    // check header
    char magic[4];
    r.read_bytes(magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0) LLM_ABORT("GGUF: bad magic %s", path.c_str());

    uint32_t version = r.read<uint32_t>();
    if (version == 0 || version > 3) LLM_ABORT("GGUF: unsupported version %u", version);

    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv      = r.read<uint64_t>();

    // read metadata KV
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.read_string();
        GGUFType type = (GGUFType)r.read<uint32_t>();

        GGUFValue val;
        if (type == GGUFType::ARRAY) {
            GGUFType elem = (GGUFType)r.read<uint32_t>();
            uint64_t n = r.read<uint64_t>();
            val.is_array = true;
            read_value(r, elem, n, val);
        } else {
            read_value(r, type, 1, val);
        }
        meta_[std::move(key)] = std::move(val);
    }

    // alignment
    uint64_t alignment = get_int("general.alignment", 32);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) LLM_ABORT("GGUF: invalid alignment");

    // build tensor table
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = r.read_string();

        uint32_t n_dims = r.read<uint32_t>();
        if (n_dims < 1 || n_dims > 4) LLM_ABORT("GGUF: bad tensor rank %s", t.name.c_str());

        uint64_t ne[4] = {1, 1, 1, 1};
        for (uint32_t j = 0; j < n_dims; ++j) ne[j] = r.read<uint64_t>();

        uint32_t gtype = r.read<uint32_t>();
        t.type = (DType)gtype;
        if (t.type != DType::F32  && t.type != DType::F16  && t.type != DType::Q8_0 &&
            t.type != DType::Q4_K && t.type != DType::Q5_K && t.type != DType::Q6_K) {
            LLM_ABORT("GGUF: unsupported tensor type %u", gtype);
        }

        t.offset = (size_t)r.read<uint64_t>();

        // flatten to 2D
        t.ne0 = (int)ne[0];
        t.ne1 = (int)(ne[1] * ne[2] * ne[3]);

        // row elements must be a multiple of block size
        type_traits ti = type_info(t.type);
        if (t.ne0 % ti.elems_per_block != 0) LLM_ABORT("GGUF: tensor row not multiple of block size %s", t.name.c_str());

        tensors_.push_back(t);
    }

    // build name -> index map
    for (size_t i = 0; i < tensors_.size(); ++i) {
        tensor_index_[tensors_[i].name] = i;
    }

    // get tensor offsets
    data_offset_ = (r.pos() + alignment - 1) & ~(alignment - 1);
    for (auto& t : tensors_) {
        t.data = base + data_offset_ + t.offset;
    }

    // metadata is parsed into C++ objects, header pages can be released
    if (data_offset_ > 0) madvise(base, data_offset_, MADV_DONTNEED);
}

} // namespace llm
