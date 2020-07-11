#include <iostream>
#include "w2l.h"

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
    } else if (strcmp(argv[1], "unpack") == 0) {
        /*
        if (argc < 4) {
            usage();
            return 1;
        }
        auto reader = b2l::Reader::open_file(argv[2]);
        file = b2l::File::read_from(reader);
        std::cout << file.to_str() << "\n";

        for (auto &section : file.sections) {
            std::cout << section.to_str() << "\n";
            std::cout << section.array().to_str() << "\n";
        }
        */
    } else {
        usage();
        return 1;
    }
}
