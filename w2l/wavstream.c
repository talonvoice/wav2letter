#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "w2lstream.h"

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        printf("Usage:  cat file.wav | %s <feature_model> <acoustic_model> <token_file>\n", argv[0]);
        return 1;
    }

    int chunk_size = 1000 * 16000 / 500;
    w2lstream *stream = w2lstream_new(argv[1], argv[2], argv[3], chunk_size);

    int16_t *buffer = calloc(chunk_size * sizeof(int16_t), 1);
    float *samples = calloc(chunk_size * sizeof(float), 1);
    int wav_header_size = 44;

    fread(buffer, wav_header_size, 1, stdin);
    while (!feof(stdin)) {
        size_t size = fread(buffer, 1, chunk_size * 2, stdin);
        if (size == 0)
            break;
        for (ssize_t i = 0; i < size / 2; i++) {
            samples[i] = buffer[i] / 32768.0;
        }
        char *text = w2lstream_run(stream, 1, samples, size / 2);
        if (strlen(text) > 0) {
            if (text[0] == ' ') {
                printf("%s ", text + 1);
            } else {
                printf("%s", text);
            }
            fflush(stdout);
        }
        free(text);
    }
    printf("\n");
    w2lstream_free(stream);
}
