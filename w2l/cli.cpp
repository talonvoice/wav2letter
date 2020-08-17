#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "w2l_encode.h"
#include "b2l.h"

void usage() {
    // TODO: w2l stream/emit/decode subcommands using samples piped to stdin like wavstream?
    std::cout << "Usage: w2l pack   <outfile> <am> <tokens> [spm.bin]" << std::endl;
    std::cout << "Usage: w2l unpack <infile>  <outdir>" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    if (strcmp(argv[1], "pack") == 0) {
        if (argc < 5) {
            usage();
            return 1;
        }
        w2l_engine *engine = w2l_engine_new();
        if (!w2l_engine_load_w2l(engine, argv[3], argv[4])) {
            printf("failed to load model: %s\n", argv[4]);
            return 1;
        }
        if (!w2l_engine_export_b2l(engine, argv[2])) {
            printf("failed to export model: %s\n", argv[2]);
        }
        // pack spm.bin into the model if provided
        if (argc == 6) {
            std::ifstream spm_file(argv[5], std::ios::binary);
            auto spm_data = std::vector<uint8_t>(std::istream_iterator<uint8_t>(spm_file), std::istream_iterator<uint8_t>());

            auto file = b2l::File::open_file(argv[2], true);
            file.add_section("spm").data(spm_data);

            auto writer = b2l::Writer::open_file(argv[2]);
            file.write_to(writer);
        }
    } else if (strcmp(argv[1], "unpack") == 0) {
        if (argc != 4) {
            usage();
            return 1;
        }
        w2l_engine *engine = w2l_engine_new();
        if (!w2l_engine_load_b2l(engine, argv[2])) {
            printf("failed to load model: %s\n", argv[4]);
            return 1;
        }
#ifdef _WIN32
        std::string sep = "\\";
#else
        std::string sep = "/";
#endif
        std::string dir = argv[3];
        std::string am_path = dir + sep + "acoustic.bin";
        std::string spm_path = dir + sep + "spm.bin";
        std::string tokens_path = dir + sep + "tokens.txt";
        if (!w2l_engine_export_w2l(engine, am_path.c_str())) {
            printf("failed to export model: %s\n", argv[2]);
            return 1;
        }
        auto model = b2l::File::open_file(argv[2]);
        if (model.has_section("spm")) {
            auto spm = model.section("spm").data();
            std::ofstream file(spm_path, std::ios::binary);
            file.write((char *)spm.data(), spm.size());
        }
        auto tokens = model.section("tokens").utf8();
        std::ofstream file(tokens_path);
        file << tokens;
    } else {
        usage();
        return 1;
    }
    return 0;
}
