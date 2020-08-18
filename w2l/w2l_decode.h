#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "w2l_common.h"

// this structure, including transitions and criterion, are copied by the callee and do not need to be retained by the caller
typedef struct {
    // encoder parameters
    float *transitions;
    size_t transitions_len;
    const char *criterion;

    // pure decode options
    int beamsize;
    int beamsizetoken;
    float beamthresh;
    float lmweight;
    float wordscore;
    float unkweight;
    bool logadd;
    float silweight;

    // DFA options

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
} w2l_decode_options;

extern w2l_decode_options w2l_decode_defaults;

typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_decoderesult w2l_decoderesult;

// int w2l_beam_count(w2l_decoder_state *state);
// char *w2l_beam_words(w2l_decoder_state *state, int beam);
// char *w2l_beam_tokens(w2l_decoder_state *state, int beam);
// float w2l_beam_score(w2l_decoder_state *state, int beam);

// FIXME: DFA, resumable decoder

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

// to use the decoder:
//  decoder = w2l_decoder_new()
//  if (!w2l_decoder_load_trie(decoder, trie_path)) {
//    if (!w2l_decoder_make_trie(decoder, trie_path)) {
//      // handle error
//    }
//    if (!w2l_decoder_load_trie(decoder, trie_path)) {
//      // handle error
//    }
//  }
//  result = w2l_decoder_decode(decoder, emission)
//  char *text = w2l_decoder_result_words(decoder, result)
//  free(text)
//  w2l_decoderesult_free(result);
//  w2l_decoder_free(decoder)

w2l_decoder *w2l_decoder_new(const char *tokens, const char *kenlm_model_path, const char *lexicon_path, const w2l_decode_options *opts);
bool w2l_decoder_load_trie(w2l_decoder *decoder, const char *trie_path);
bool w2l_decoder_make_trie(w2l_decoder *decoder, const char *trie_path);

w2l_decoderesult *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission);
char *w2l_decoder_result_words(w2l_decoder *decoder, w2l_decoderesult *decoderesult);
char *w2l_decoder_result_tokens(w2l_decoder *decoder, w2l_decoderesult *decoderesult);
void w2l_decoderesult_free(w2l_decoderesult *decoderesult);
void w2l_decoder_free(w2l_decoder *decoder);

/** Decode emisssions according to dfa model, return decoded text.
 *
 * If the decode fails or no good paths exist the result will be NULL.
 * If it is not null, the caller is responsible for free()ing the string.
 *
 * The dfa argument points to the first w2l_dfa_node. It is expected that
 * its address and edge offsets can be used to traverse the full dfa.
 */
char *w2l_decoder_dfa(w2l_decoder *decoder, w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size);

// greedy-decode an emission (e.g. viterbi or argmax)
char *w2l_decoder_greedy(w2l_decoder *decoder, w2l_emission *emission);

#ifdef __cplusplus
} // extern "C"
#endif
