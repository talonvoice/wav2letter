#include <cstring>
#include <iostream>
#include <vector>
#include "w2l.h"
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
        if (argc != 3) {
            usage();
            return 1;
        }
        w2l_engine *engine = w2l_engine_new();
        if (!w2l_engine_load_b2l(engine, argv[3])) {
            printf("failed to load model: %s\n", argv[4]);
            return 1;
        }
        if (!w2l_engine_export_w2l(engine, argv[2])) {
            printf("failed to export model: %s\n", argv[2]);
            return 1;
        }
    } else {
        usage();
        return 1;
    }
    return 0;
}
