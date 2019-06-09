#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_emission w2l_emission;

w2l_engine *w2l_engine_new(const char *acoustic_model_path, const char *tokens_path);
w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count);
void w2l_engine_free(w2l_engine *engine);

char *w2l_emission_text(w2l_emission *emission);
void w2l_emission_free(w2l_emission *emission);

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path);
char *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission);
void w2l_decoder_free(w2l_decoder *decoder);

#ifdef __cplusplus
} // extern "C"
#endif
