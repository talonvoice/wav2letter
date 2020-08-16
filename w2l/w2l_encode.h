#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdbool.h>

// malloc(sizeof(w2l_emission) + sizeof(float) * n_nrames * n_tokens);
typedef struct w2l_emission {
    int n_frames;
    int n_tokens;
    float matrix[0];
} w2l_emission;

typedef struct w2l_engine w2l_engine;

// encoder
w2l_engine *w2l_engine_new();
bool w2l_engine_load_w2l(w2l_engine *engine, const char *acoustic_model_path, const char *tokens_path);
bool w2l_engine_load_b2l(w2l_engine *engine, const char *path);
bool w2l_engine_export_w2l(w2l_engine *engine, const char *path);
bool w2l_engine_export_b2l(w2l_engine *engine, const char *path);
void w2l_engine_free(w2l_engine *engine);

// emission based stuff
w2l_emission *w2l_engine_forward(w2l_engine *engine, float *samples, size_t sample_count);
// free(emission);

#ifdef __cplusplus
} // extern "C"
#endif
