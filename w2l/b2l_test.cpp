#include "b2l.h"
#include <functional>

static bool _test_failed = false;
static int _pass_count = 0;
static int _total_count = 0;
void _run(std::string name, std::function<void ()> fn) {
    _total_count++;
    _test_failed = false;
    std::cerr << "[+] RUN " << name << "() ";
    fn();
    if (_test_failed) {
        std::cerr << "[FAIL]" << std::endl;
    } else {
        _pass_count++;
        std::cerr << "[PASS]" << std::endl;
    }
}
#define run(name) _run(#name, name)

void named_assert(std::string name, bool expr) {
    if (!expr) {
        std::cerr << "\nassertion failed: " << name;
        _test_failed = true;
    }
}
#undef assert
#define assert(expr) named_assert(#expr, expr)

void hexdump(std::vector<uint8_t> data) {
    for (auto b : data) {
        printf("%02x", b);
    }
    printf("\n");
}

void assert_equal(std::string a, std::string b) {
    std::ostringstream ostr;
    ostr << "string a \"" << a << "\" == b \"" << b << "\"";
    named_assert(ostr.str(), a == b);
}

void assert_equal(b2l::Section &a, b2l::Section &b) {
    assert(a.name == b.name);
    assert(a.desc == b.desc);
    assert(a.type == b.type);
    assert(a.as_bytes() == b.as_bytes());
}

void assert_equal(b2l::File &a, b2l::File &b) {
    assert(a.name == b.name);
    assert(a.sections.size() == b.sections.size());
    if (a.sections.size() == b.sections.size()) {
        for (ssize_t i = 0; i < a.sections.size(); i++) {
            assert_equal(a.sections[i], b.sections[i]);
        }
    }
}

b2l::File assert_file_round_trip(b2l::File &file1) {
    auto bytes = file1.as_bytes();
    auto file2 = b2l::File::from_bytes(bytes);
    assert_equal(file1, file2);
    return file2;
}

#pragma mark tests

void test_section() {
    b2l::Section section("name");
    section.utf8("test");
    assert_equal(section.utf8(), "test");
}

void test_file_no_sections() {
    b2l::File file1;
    assert_file_round_trip(file1);
}

void test_file_config() {
    b2l::File file1;
    auto &config = file1.add_section("config");
    config.keyval({
        {"name", "example"},
        {"criterion", "ctc"},
        {"quantization", ""},
    });
    auto file2 = assert_file_round_trip(file1);
    auto &config2 = file2.section("config");

    auto kv1 = config.keyval();
    auto kv2 = config2.keyval();

    assert_equal(kv1["name"], "example");
    assert_equal(kv1["criterion"], "ctc");
    assert_equal(kv1["quantization"], "");

    assert_equal(kv1["name"], kv2["name"]);
    assert_equal(kv1["criterion"], kv2["criterion"]);
    assert_equal(kv1["quantization"], kv2["quantization"]);
}

void test_file_simple() {
    b2l::File file1("name1");
    auto &section1 = file1.add_section("section name");
    std::vector<float> array1(1000);
    std::iota(array1.begin(), array1.end(), 1);
    section1.array(array1);
    assert_file_round_trip(file1);
}

void test_file_layers() {
    b2l::File file1("name1");
    auto &section = file1.add_section("layer_section");
    std::vector<b2l::Layer> layers;
    layers.emplace_back(b2l::Layer{"L1", {std::vector<float>{1.1, 2.2, 3.3}}});
    layers.emplace_back(b2l::Layer{"L2", {std::vector<int>{4, 5, 6}}});
    layers.emplace_back(b2l::Layer{"L3", {std::vector<double>{7.0, 8.4, 9.1}}});
    section.layers(layers);
    auto file2 = assert_file_round_trip(file1);

    auto layers1 = file1.section("layer_section").layers();
    auto layers2 = file2.section("layer_section").layers();
    assert(layers1.size() == layers2.size());
    for (ssize_t i = 0; i < std::min(layers1.size(), layers2.size()); i++) {
        auto &layer1 = layers1[i];
        auto &layer2 = layers2[i];
        assert_equal(layer1.arch, layer2.arch);
        assert(std::abs(layer1.scale - layer2.scale) < 1e-9);
        assert(layer1.offset == layer2.offset);
        assert(layer1.params.size() == layer2.params.size());
        for (ssize_t j = 0; j < std::min(layer1.params.size(), layer2.params.size()); j++) {
            auto &param1 = layer1.params[j];
            auto &param2 = layer2.params[j];
            assert(param1.size() == param2.size());
            uint8_t *data1 = nullptr, *data2 = nullptr;
            size_t size1 = 0, size2 = 0;
            param1.raw(&data1, &size1);
            param2.raw(&data2, &size2);
            assert(size1 != 0 && size2 != 0 && size1 == size2);
            assert(size1 == size2 && memcmp(data1, data2, size1) == 0);
            assert_equal(param1.to_str(), param2.to_str());
        }
    }
}

void test_file_all_section_types() {
    b2l::File file1("filename");
    auto &data = file1.add_section("data");
    std::vector<uint8_t> data_value{1, 2, 3};
    data.data(data_value);

    auto &utf8 = file1.add_section("utf8");
    utf8.utf8("test string");

    auto &keyval = file1.add_section("keyval");
    std::map<std::string, std::string> keyval_map;
    keyval_map["a"] = "b";
    keyval_map["another key"] = "another key";
    keyval_map["more keys"] = "more values";
    keyval.keyval(keyval_map);

    auto &array_float = file1.add_section("array_float");
    std::vector<float> array_float_value{4, 3, 2, 1};
    array_float.array(array_float_value);

    auto &layers = file1.add_section("layers");
    std::vector<b2l::Layer> layer_list;
    layer_list.emplace_back(b2l::Layer{"L1", {std::vector<float>{1.1, 2.2, 3.3}}});
    layer_list.emplace_back(b2l::Layer{"L2", {std::vector<int>{4, 5, 6}}});
    layer_list.emplace_back(b2l::Layer{"L3", {std::vector<double>{7.0, 8.4, 9.1}}});
    layers.layers(layer_list);

    auto &config = file1.add_section("config");
    config.keyval({
        {"name", "example"},
        {"criterion", "ctc"},
        {"quantization", ""},
    });

    auto &str2 = file1.add_section("str2");
    str2.utf8("test string 2");

    assert_file_round_trip(file1);

    auto writer = b2l::Writer::open_file("out.b2l");
    file1.write_to(writer);
    std::cout << file1.to_str() << "\n";
}

#pragma mark end tests

int main(int argc, char **argv) {
    run(test_section);
    run(test_file_no_sections);
    run(test_file_config);
    run(test_file_simple);
    run(test_file_layers);
    run(test_file_all_section_types);

    std::cerr << "    (" << _pass_count << "/" << _total_count << ") tests passed" << std::endl;
    return (_pass_count != _total_count);
}
