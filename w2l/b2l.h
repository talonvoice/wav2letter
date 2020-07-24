// #pragma once
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace b2l {

struct Array {
    struct Type {
        typedef enum {NONE = 0, FP16 = 1, FP32 = 2, FP64 = 3,
                      I8 = 4, I16 = 5, I32 = 6, I64 = 7} Enum;
        static std::string to_str(Enum type) {
            switch (type) {
                case FP16: return "fp16";
                case FP32: return "fp32";
                case FP64: return "fp64";
                case I8:   return "i8";
                case I16:  return "i16";
                case I32:  return "i32";
                case I64:  return "i64";
                default:   return "";
            }
        }
        static Enum from_string(std::string name) {
            if (name == "fp16") return FP16;
            if (name == "fp32") return FP32;
            if (name == "fp64") return FP64;
            if (name == "i8")   return I8;
            if (name == "i16")  return I16;
            if (name == "i32")  return I32;
            if (name == "i64")  return I64;
            return NONE;
        }
        // TODO: FP16
        static Enum from_value(const std::vector<float>   &value) { return FP32; }
        static Enum from_value(const std::vector<double>  &value) { return FP64; }
        static Enum from_value(const std::vector<int8_t>  &value) { return I8;   }
        static Enum from_value(const std::vector<int16_t> &value) { return I16;  }
        static Enum from_value(const std::vector<int32_t> &value) { return I32;  }
        static Enum from_value(const std::vector<int64_t> &value) { return I64;  }
    };

    Array() : _type(Type::NONE), _value(nullptr) {}
    template <typename T>
    Array(std::vector<T> &value) : _type(Type::NONE), _value(nullptr) {
        this->array(std::move(value));
    }
    template <typename T>
    Array(std::vector<T> value) : _type(Type::NONE), _value(nullptr) {
        this->array(std::move(value));
    }
    ~Array() { this->delete_value(); }
    Array(const Array &old) : _type(Type::NONE), _value(nullptr) { this->copy_value_from(old); }
    Array(Array &&other) {
        _type  = other._type;
        _value = other._value;
        other._type  = Type::NONE;
        other._value = nullptr;
    }
    Array &operator=(Array &&other) {
        if (&other != this) {
            this->delete_value();
            _type  = other._type;
            _value = other._value;
            other._type  = Type::NONE;
            other._value = nullptr;
        }
        return *this;
    }
    Array &operator=(const Array &other) {
        this->copy_value_from(other);
        return *this;
    }

    void init(std::string type, size_t length) {
        this->delete_value();
        this->_type = Type::from_string(type);
        switch (this->_type) {
            // TODO: FP16
            case Type::FP32: this->init_type<float  >(length); break;
            case Type::FP64: this->init_type<double >(length); break;
            case Type::I8:   this->init_type<int8_t >(length); break;
            case Type::I16:  this->init_type<int16_t>(length); break;
            case Type::I32:  this->init_type<int32_t>(length); break;
            case Type::I64:  this->init_type<int64_t>(length); break;
            default: break;
        }
    }

    void raw(uint8_t **data_out, size_t *byte_size_out) {
        switch (this->_type) {
            // TODO: FP16
            case Type::FP32: this->raw_type<float  >(data_out, byte_size_out); break;
            case Type::FP64: this->raw_type<double >(data_out, byte_size_out); break;
            case Type::I8:   this->raw_type<int8_t >(data_out, byte_size_out); break;
            case Type::I16:  this->raw_type<int16_t>(data_out, byte_size_out); break;
            case Type::I32:  this->raw_type<int32_t>(data_out, byte_size_out); break;
            case Type::I64:  this->raw_type<int64_t>(data_out, byte_size_out); break;
            default: *data_out = nullptr; *byte_size_out = 0; break;
        }
    }

    Type::Enum type() { return _type; }
    std::string type_str() { return Type::to_str(this->_type); }

    size_t size() {
        switch (this->_type) {
            // TODO: FP16
            case Type::FP32: return this->array<float  >().size();
            case Type::FP64: return this->array<double >().size();
            case Type::I8:   return this->array<int8_t >().size();
            case Type::I16:  return this->array<int16_t>().size();
            case Type::I32:  return this->array<int32_t>().size();
            case Type::I64:  return this->array<int64_t>().size();
            default: return 0;
        }
    }

    std::string to_str() {
        switch (this->_type) {
            // TODO: FP16
            case Type::FP32: return this->to_str_type<float  >();
            case Type::FP64: return this->to_str_type<double >();
            case Type::I8:   return this->to_str_type<int8_t >();
            case Type::I16:  return this->to_str_type<int16_t>();
            case Type::I32:  return this->to_str_type<int32_t>();
            case Type::I64:  return this->to_str_type<int64_t>();
            default: return "";
        }
    }

    template <typename T>
    std::vector<T> &array() {
        std::vector<T> &value = *reinterpret_cast<std::vector<T> *>(this->_value);
        Type::Enum type = Type::from_value(value);
        if (type != this->_type) {
            throw std::runtime_error("b2l::Array.array() type mismatch");
        }
        return value;
    }

    template <typename T>
    std::vector<T> move_array() {
        std::vector<T> &value = *reinterpret_cast<std::vector<T> *>(this->_value);
        Type::Enum type = Type::from_value(value);
        if (type != this->_type) {
            throw std::runtime_error("b2l::Array.array() type mismatch");
        }
        std::vector<T> tmp = std::move(value);
        this->delete_value();
        return tmp;
    }

    template <typename T>
    void array(const std::vector<T> &value) {
        this->delete_value();
        this->_type = Type::from_value(value);
        this->_value = new std::vector<T>(value);
    }
private:
    template <typename T>
    std::string to_str_type() {
        std::ostringstream ostr;
        ostr << "{";
        const auto &array = this->array<T>();
        for (auto it = array.begin(); it < array.end(); it++) {
            ostr << *it;
            if (it != array.end() - 1) {
                ostr << ", ";
            }
        }
        ostr << "}";
        return ostr.str();
    }

    template <typename T>
    void init_type(size_t length) {
        auto value = new std::vector<T>(length);
        this->_value = value;
    }

    template <typename T>
    void raw_type(uint8_t **data_out, size_t *byte_size_out) {
        auto &value = this->array<T>();
        *data_out = (uint8_t *)&value[0];
        *byte_size_out = sizeof(T) * value.size();;
    }

    void copy_value_from(const Array &old) {
        if (!old._value || old._type == Type::NONE) {
            this->_value = nullptr;
            this->_type = Type::NONE;
            return;
        }
        switch (old._type) {
            // TODO: FP16
            case Type::FP32: this->copy_value<float  >(old); break;
            case Type::FP64: this->copy_value<double >(old); break;
            case Type::I8:   this->copy_value<int8_t >(old); break;
            case Type::I16:  this->copy_value<int16_t>(old); break;
            case Type::I32:  this->copy_value<int32_t>(old); break;
            case Type::I64:  this->copy_value<int64_t>(old); break;
            default: break;
        }
    }
    template <typename T>
    void copy_value(const Array &old) {
        auto &value = const_cast<Array &>(old).array<T>();
        this->array<T>(value);
    }

    void delete_value() {
        if (!this->_value) return;
        switch (this->_type) {
            // TODO: FP16
            case Type::FP32: delete reinterpret_cast<std::vector<float  > *>(this->_value); break;
            case Type::FP64: delete reinterpret_cast<std::vector<double > *>(this->_value); break;
            case Type::I8:   delete reinterpret_cast<std::vector<int8_t > *>(this->_value); break;
            case Type::I16:  delete reinterpret_cast<std::vector<int16_t> *>(this->_value); break;
            case Type::I32:  delete reinterpret_cast<std::vector<int32_t> *>(this->_value); break;
            case Type::I64:  delete reinterpret_cast<std::vector<int64_t> *>(this->_value); break;
            default: break;
        }
        this->_value = nullptr;
    }

private:
    Type::Enum _type;
    void *_value;
};

class Reader {
public:
    // constructors
    Reader(std::shared_ptr<std::istream> stream) : f(stream) {}
    static Reader open_memory() {
        return Reader(std::make_shared<std::stringstream>());
    }
    static Reader open_memory(std::string buffer) {
        return Reader(std::make_shared<std::stringstream>(std::move(buffer)));
    }
    static Reader open_memory(std::vector<uint8_t> buffer) {
        std::string strbuf(buffer.begin(), buffer.end());
        return open_memory(std::move(strbuf));
    }
    static Reader open_file(std::string path) {
        return Reader(std::make_shared<std::ifstream>(path, std::ios::in | std::ios::binary));
    }

    // methods
    void seek(ssize_t pos) { f->seekg(pos); }
    ssize_t tell()         { return f->tellg(); }
    bool eof()             { return f->eof(); }
    
    uint8_t read8() {
        char byte = 0;
        f->read(&byte, 1);
        assert_gcount(1);
        return (uint8_t)byte;
    }

    uint64_t read64() {
        uint64_t n = 0;
        uint8_t bytes[8] = {0};
        read_bytes(bytes, 8);
        for (size_t i = 0; i < 8; i++) {
            n |= ((uint64_t)bytes[i] << (i * 8LLU));
        }
        return n;
    }

    float fp32() {
        float n = 0;
        f->read((char *)&n, 4);
        return n;
    }

    void read_bytes(uint8_t *data, size_t size) {
        f->read((char *)data, size);
        assert_gcount(size);
    }

    void read_bytes(std::vector<uint8_t> &data) {
        if (data.empty()) return;
        read_bytes(&data[0], data.size());
    }

    std::vector<uint8_t> read_bytes(size_t size) {
        std::vector<uint8_t> tmp(size);
        read_bytes(tmp);
        return tmp;
    }

    std::string short_string() {
        uint8_t size = read8();
        std::vector<uint8_t> tmp = read_bytes(size);
        assert_gcount(size);
        return std::string(tmp.begin(), tmp.end());
    }

    std::string long_string() {
        uint64_t size = read64();
        std::vector<uint8_t> tmp = read_bytes(size);
        assert_gcount(size);
        return std::string(tmp.begin(), tmp.end());
    }

    // TODO:
    Array array() {
        Array tmp;
        std::string type = short_string();
        size_t length = read64();
        uint8_t *data;
        size_t byte_size;
        tmp.init(type, length);
        tmp.raw(&data, &byte_size);
        if (byte_size > 0) {
            read_bytes(data, byte_size);
        }
        return tmp;
    }
    template <typename T>
    std::vector<T> array() {
        std::vector<T> value;
        auto type = Array::Type::to_str(Array::Type::from_value(value));
        if (type != short_string()) {
            throw std::runtime_error("b2l::Reader: array type mismatch");
        }
        value.resize(read64());
        if (value.size() > 0) {
            read_bytes((uint8_t *)&value[0], sizeof(T) * value.size());
        }
        return value;
    }
private:
    void assert_gcount(size_t size) {
        if (f->gcount() < size) {
            throw std::runtime_error("b2l::Reader unexpected end of stream");
        }
    }

    std::shared_ptr<std::istream> f;
};

// TODO: surface errors? maybe exceptions?
class Writer {
public:
    // constructors
    Writer(std::shared_ptr<std::ostream> stream) : f(stream) {}
    static Writer open_memory() {
        return Writer(std::make_shared<std::stringstream>());
    }
    static Writer open_file(std::string path) {
        return Writer(std::make_shared<std::ofstream>(path, std::ios::out | std::ios::binary));
    }

    // methods
    void write8(uint8_t b) {
        char byte = (char)b;
        f->write(&byte, 1);
    }

    void write64(uint64_t n) {
        char bytes[8] = {0};
        for (ssize_t i = 0; i < 8; i++) {
            bytes[i] = (n >> (i * 8LL)) & 0xff;
        }
        f->write(bytes, 8);
    }

    void fp32(float n) {
        char bytes[4] = {0};
        memcpy(bytes, &n, 4);
        f->write(bytes, 4);
    }

    void write_bytes(const uint8_t *data, size_t size) {
        f->write((char *)data, size);
    }

    void write_bytes(const std::vector<uint8_t> &data) {
        if (data.empty()) return;
        write_bytes(&data[0], data.size());
    }

    void short_string(const std::string &str) {
        write8(str.size());
        return write_bytes((const uint8_t *)str.data(), std::min((size_t)255, str.size()));
    }

    void long_string(const std::string &str) {
        write64(str.size());
        return write_bytes((const uint8_t *)str.data(), str.size());
    }

    void array(Array &value) {
        short_string(value.type_str());
        write64(value.size());
        uint8_t *data;
        size_t byte_size;
        value.raw(&data, &byte_size);
        if (byte_size > 0) {
            write_bytes(data, byte_size);
        }
    }
    template <typename T>
    void array(std::vector<T> &value) {
        auto type = Array::Type::to_str(Array::Type::from_value(value));
        short_string(type);
        write64(value.size());
        if (value.size() > 0) {
            write_bytes((uint8_t *)&value[0], sizeof(T) * value.size());
        }
    }
private:
    std::shared_ptr<std::ostream> f;
};

struct Layer {
    Layer() {}
    Layer(std::string arch, std::vector<Array> params) {
        this->arch = std::move(arch);
        this->params = params;
    }

    std::string arch;
    std::vector<Array> params;
    float scale = 1.0;
    int64_t offset = 0;
};

struct Section {
public:
    // static const std::unordered_set<std::string> Types = {"utf8", "keyval", "data", "array", "params"};

public:
    Section(std::string name="") :
            name(name),
            stream(std::make_shared<std::stringstream>()),
            _reader(nullptr),
            _writer(nullptr) {
        _wrote = false;
        _off = 0;
        _size = 0;
        _end = 0;
        _reader = Reader(stream);
        _writer = Writer(stream);
    }

    void set_reader(Reader reader, ssize_t off, size_t size) {
        _reader = reader;
        _off = off;
        _size = size;
        _end = off + size;
    }

    Reader &reader() {
        if (_wrote) {
            _reader = Reader(stream);
            _off = 0;
            _size = stream->str().size();
            _end = _off + _size;
            _wrote = false;
        }
        _reader.seek(_off);
        return _reader;
    }

    Writer &writer() {
        _wrote = true;
        return _writer;
    }

    static Section read_from(Reader &reader, bool read_all=false) {
        std::string name = reader.short_string();
        std::string type = reader.short_string();
        std::string desc = reader.long_string();
        size_t size = reader.read64();

        Section out;
        out.name = std::move(name);
        out.type = std::move(type);
        out.desc = std::move(desc);

        size_t pos = reader.tell();
        if (read_all) {
            auto data = reader.read_bytes(size);
            out.set_reader(Reader::open_memory(data), 0, size);
        } else {
            out.set_reader(reader, pos, size);
        }
        reader.seek(pos + size);
        return out;
    }

    void write_to(Writer &writer) {
        if (!stream || type == "") {
            throw std::runtime_error("cannot write empty section");
        }
        auto data = this->as_bytes();
        writer.short_string(name);
        writer.short_string(type);
        writer.long_string(desc);
        writer.write64(data.size());
        writer.write_bytes(data);
    }

    std::vector<uint8_t> as_bytes() {
        return this->reader().read_bytes(_size);
    }

    // section.data()
    std::vector<uint8_t> data() {
        this->assert_type("data");
        return this->as_bytes();
    }
    void data(std::vector<uint8_t> &bytes) {
        type = "data";
        writer().write_bytes(bytes);
    }

    // section.utf8()
    std::string utf8() {
        this->assert_type("utf8");
        auto data = this->reader().read_bytes(_size);
        return std::string(data.begin(), data.end());
    }
    void utf8(std::string value) {
        type = "utf8";
        writer().write_bytes((uint8_t *)&value[0], value.size());
    }

    // section.keyval()
    std::unordered_map<std::string, std::string> keyval() {
        this->assert_type("keyval");
        auto reader = this->reader();
        std::unordered_map<std::string, std::string> tmp;
        while (reader.tell() < _end && !reader.eof()) {
            std::string key = reader.short_string();
            std::string value = reader.long_string();
            if (key != "") {
                tmp[key] = value;
            }
        }
        return tmp;
    }
    void keyval(std::unordered_map<std::string, std::string> &map) {
        type = "keyval";
        auto writer = this->writer();
        for (auto &pair : map) {
            writer.short_string(pair.first);
            writer.long_string(pair.second);
        }
    }
    void keyval(std::initializer_list<std::pair<std::string, std::string>> init) {
        std::unordered_map<std::string, std::string> map(init.begin(), init.end());
        keyval(map);
    }

    // section.array<float>()
    template<typename T>
    std::vector<T> array() {
        this->assert_type("array");
        return this->reader().array<T>();
    }
    Array array() {
        this->assert_type("array");
        return this->reader().array();
    }
    template<typename T>
    void array(std::vector<T> &values) {
        type = "array";
        writer().array(values);
    }

    // for (auto &layer : section.layers()) {
    //   layer.name
    //   layer.scale
    //   layer.offset
    //   layer.params[0].array<float>();
    std::vector<Layer> layers() {
        this->assert_type("layers");
        auto reader = this->reader();
        ssize_t count = reader.read64();
        std::vector<Layer> layers;
        layers.resize(count);
        for (auto &layer : layers) {
            layer.arch = reader.long_string();
            layer.scale = reader.fp32();
            layer.offset = (int64_t)reader.read64();
            auto param_count = reader.read64();
            layer.params.resize(param_count);
            for (ssize_t i = 0; i < param_count; i++) {
                layer.params[i] = std::move(reader.array());
            }
        }
        return layers;
    }
    void layers(std::vector<Layer> &layers) {
        type = "layers";
        auto writer = this->writer();
        writer.write64(layers.size());
        for (auto &layer : layers) {
            writer.long_string(layer.arch);
            writer.fp32(layer.scale);
            writer.write64((uint64_t)layer.offset);
            writer.write64(layer.params.size());
            for (auto &array : layer.params) {
                writer.array(array);
            }
        }
    }

    std::string to_str() const {
        std::ostringstream ostr;
        size_t size = 0;
        if (stream) {
            size = stream->tellp();
        } else {
            size = _size;
        }
        ostr << "{section type=" << type << " name=" << name << " desc=" << desc << " size=" << size << "}";
        return ostr.str();
    }
private:
    void assert_type(std::string type) {
        if (this->type != type) {
            std::ostringstream ostr;
            ostr << "b2l::Section type mismatch: called " << type << "() but actual type is " << this->type << "()";
            throw std::runtime_error(ostr.str());
        }
    }

public:
    std::string name;
    std::string desc;
    std::string type;
private:
    Reader _reader;
    Writer _writer;
    ssize_t _off;
    size_t _size;
    size_t _end;
    bool _wrote;
    std::shared_ptr<std::stringstream> stream;
};

struct File {
public:
    std::string magic;        // "BW2L"
    uint8_t version;          // 1
    std::string name;         // "streaming-convnets"
    std::vector<Section> sections;
    std::unordered_map<std::string, int> section_lookup;

    File(std::string name="") : magic("BW2L"), version(1), name(name) {}

    bool has_section(std::string name) {
        auto idx = section_lookup.find(name);
        return idx != section_lookup.end();
    }

    Section &section(std::string name) {
        auto idx = section_lookup.find(name);
        if (idx == section_lookup.end()) {
            throw std::out_of_range("section not found");
        }
        return sections[idx->second];
    }

    Section &add_section(std::string name) {
        sections.emplace_back(Section(name));
        section_lookup[name] = sections.size() - 1;
        return sections.back();
    }

    std::string to_str() const {
        std::ostringstream ostr;
        ostr << "{b2l::File magic='" << magic << "' version=" << int(version) << " name='" << name << "' section_count=" << sections.size() << "}";
        return ostr.str();
    }

    static File read_from(Reader &reader, bool read_all=false) {
        File out;
        auto data = reader.read_bytes(4);
        out.magic = std::string(data.begin(), data.end());
        out.version = reader.read8();
        if (out.magic != "BW2L" || out.version != 1) {
            throw std::runtime_error("Unsupported file magic or version");
        }
        out.name = reader.short_string();
        out.sections.resize(reader.read64());
        for (ssize_t i = 0; i < out.sections.size(); i++) {
            auto section = out.sections[i] = Section::read_from(reader, read_all);
            out.section_lookup[section.name] = i;
        }
        return out;
    }

    static File open_file(std::string path, bool read_all=false) {
        auto reader = Reader::open_file(path);
        return File::read_from(reader, read_all);
    }

    void write_to(Writer &writer) {
        writer.write_bytes((uint8_t *)"BW2L", 4); // magic
        writer.write8(1);                         // version
        writer.short_string(name);
        writer.write64(sections.size());
        for (auto &section : sections) {
            section.write_to(writer);
        }
    }

    std::vector<uint8_t> as_bytes() {
        auto stream = std::make_shared<std::ostringstream>();
        Writer writer(stream);
        write_to(writer);
        auto str = stream->str();
        auto data = (uint8_t *)str.data();
        return std::vector<uint8_t>(data, data + str.size());
    }

    static File from_bytes(std::vector<uint8_t> &bytes) {
        std::string tmp(bytes.begin(), bytes.end());
        auto reader = b2l::Reader::open_memory(tmp);
        return File::read_from(reader);
    }
};

} // namespace b2l
