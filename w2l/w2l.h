#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdbool.h>

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_decoderesult w2l_decoderesult;
typedef struct w2l_emission w2l_emission;

typedef struct {
    int beamsize;
    int beamsizetoken;
    float beamthresh;
    float lmweight;
    float wordscore;
    float unkweight;
    bool logadd;
    float silweight;
} w2l_decode_options;

extern w2l_decode_options w2l_decode_defaults;

w2l_engine *w2l_engine_new();
bool w2l_engine_load_w2l(w2l_engine *engine, const char *acoustic_model_path, const char *tokens_path);
bool w2l_engine_load_b2l(w2l_engine *engine, const char *path);
bool w2l_engine_export_w2l(w2l_engine *engine, const char *path);
bool w2l_engine_export_b2l(w2l_engine *engine, const char *path);
w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count);
void w2l_engine_free(w2l_engine *engine);

char *w2l_emission_text(w2l_emission *emission);
float *w2l_emission_values(w2l_emission *emission, int *frames, int *tokens);
void w2l_emission_free(w2l_emission *emission);

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path, const char *flattrie_path, const w2l_decode_options *opts);
w2l_decoderesult *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission);
char *w2l_decoder_result_words(w2l_decoder *decoder, w2l_decoderesult *decoderesult);
char *w2l_decoder_result_tokens(w2l_decoder *decoder, w2l_decoderesult *decoderesult);
void w2l_decoderesult_free(w2l_decoderesult *decoderesult);
void w2l_decoder_free(w2l_decoder *decoder);

bool w2l_make_flattrie(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path, const char *flattrie_path);

#pragma pack(1)
typedef struct {
    uint16_t token;
    int32_t offset;
} w2l_dfa_edge;

typedef struct {
    uint8_t flags;
    uint16_t nEdges;
    w2l_dfa_edge edges[0];
} w2l_dfa_node;
#pragma pack()

typedef struct {
    /** Score for successfully decoded commands.
     *
     * Competes with language wordscore.
     */
    float command_score;

    /** Threshold for rejection.
     *
     * The emission-transmission score of rejection_window_frame adjacent tokens
     * divided by the score of the same area in the viterbi path. If the fraction
     * is below this threshold the decode will be rejected.
     *
     * Values around 0.55 work ok.
     */
    float rejection_threshold;

    /** Window size for decode vs viterbi comparison.
     *
     * Values around 8 make sense.
     */
    int rejection_window_frames;

    /** Whether to print debug messages to stdout. */
    bool debug;
} w2l_dfa_decode_options;

/** Decode emisssions according to dfa model, return decoded text.
 *
 * If the decode fails or no good paths exist the result will be NULL.
 * If it is not null, the caller is responsible for free()ing the string.
 *
 * The dfa argument points to the first w2l_dfa_node. It is expected that
 * its address and edge offsets can be used to traverse the full dfa.
 */
char *w2l_decoder_dfa(w2l_engine *engine, w2l_decoder *decoder, w2l_emission *emission, w2l_dfa_node *dfa, w2l_dfa_decode_options *opts);

#ifdef __cplusplus
} // extern "C"
#endif
