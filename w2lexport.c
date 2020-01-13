#include <stdio.h>
#include "w2l.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <input model> <tokens.txt> <output model>\n", argv[0]);
        return 1;
    }
    w2l_engine *engine = w2l_engine_new(argv[1], argv[2]);
    w2l_engine_export(engine, argv[3]);
    return 0;
}
