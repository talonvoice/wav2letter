#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_decoderesult w2l_decoderesult;
typedef struct w2l_emission w2l_emission;

typedef struct {
    int beamsize;
    float beamthresh;
    float lmweight;
    float wordscore;
    float unkweight;
    bool logadd;
    float silweight;
} w2l_decode_options;

static w2l_decode_options w2l_decode_defaults {
    2500,
    25,
    1.0,
    1.0,
    -INFINITY,
    false,
    0.0,
};

w2l_engine *w2l_engine_new(const char *acoustic_model_path, const char *tokens_path);
w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count);
bool w2l_engine_export(w2l_engine *engine, const char *path);
void w2l_engine_free(w2l_engine *engine);

char *w2l_emission_text(w2l_emission *emission);
float *w2l_emission_values(w2l_emission *emission, int *frames, int *tokens);
void w2l_emission_free(w2l_emission *emission);

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path, const w2l_decode_options *opts);
w2l_decoderesult *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission);
char *w2l_decoder_result_words(w2l_decoder *decoder, w2l_decoderesult *decoderesult);
char *w2l_decoder_result_tokens(w2l_decoder *decoder, w2l_decoderesult *decoderesult);
void w2l_decoderesult_free(w2l_decoderesult *decoderesult);
void w2l_decoder_free(w2l_decoder *decoder);

#pragma pack(1)
typedef struct {
    uint8_t token;
    int32_t offset;
} cfg_edge;

typedef struct {
    uint8_t flags;
    uint8_t nEdges;
    cfg_edge edges[0];
} cfg;
#pragma pack()

typedef struct {
    w2l_decode_options cmdDecodeOpts; // silweight of around 0.5 was required for me
    float rejectionThreshold; // 0.85 maybe? 0 would disable command rejection. 1 would reject everything.
    int rejectionTokenWindow; // 8
    bool debug; // stdout printing
} cmd_decode_opts;

char *w2l_decoder_dfa(w2l_engine *engine, w2l_decoder *decoder, w2l_emission *emission, cfg *dfa, cmd_decode_opts opts);

#ifdef __cplusplus
} // extern "C"
#endif
