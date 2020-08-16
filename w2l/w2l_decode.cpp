#include "w2l_decode.h"
#include "w2l_decode_backend.cpp"

using namespace w2l;

extern "C" {

w2l_decode_options w2l_decode_defaults {
    nullptr, // transitions
    0, // transitions_size
    "none", // criterion

    // decode options
    500, // beamsize
    100, // beamsizetoken
    25,  // beamthresh
    1.0, // lmweight
    1.0, // wordscore
    -INFINITY, // unkweight
    false, // logadd
    0.0,   // silweight

    // DFA decode options
    1.0,   // command_score
    0.55,  // rejection_threshold
    8,     // rejection_window_frames
    false, // debug
};

w2l_decoder *w2l_decoder_new(const char *tokens, const char *kenlm_model_path, const char *lexicon_path, const char *flattrie_path, const w2l_decode_options *opts) {
    // TODO: what other config? beam size? smearing? lm weight?
    auto decoder = new PublicDecoder(tokens, kenlm_model_path, lexicon_path, flattrie_path, opts);
    return reinterpret_cast<w2l_decoder *>(decoder);
}

w2l_decoderesult *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission) {
    auto result = new DecodeResult(reinterpret_cast<PublicDecoder *>(decoder)->decode(emission));
    return reinterpret_cast<w2l_decoderesult *>(result);
}

char *w2l_decoder_result_words(w2l_decoder *decoder, w2l_decoderesult *decoderesult) {
    auto decoderObj = reinterpret_cast<PublicDecoder *>(decoder);
    auto result = reinterpret_cast<DecodeResult *>(decoderesult);
    return decoderObj->resultWords(*result);
}

char *w2l_decoder_result_tokens(w2l_decoder *decoder, w2l_decoderesult *decoderesult) {
    auto decoderObj = reinterpret_cast<PublicDecoder *>(decoder);
    auto result = reinterpret_cast<DecodeResult *>(decoderesult);
    return decoderObj->resultTokens(*result);
}

void w2l_decoderesult_free(w2l_decoderesult *decoderesult) {
    if (decoderesult)
        delete reinterpret_cast<DecodeResult *>(decoderesult);
}

void w2l_decoder_free(w2l_decoder *decoder) {
    if (decoder)
        delete reinterpret_cast<PublicDecoder *>(decoder);
}

char *w2l_decoder_dfa(w2l_decoder *decoder, w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size) {
    return reinterpret_cast<PublicDecoder *>(decoder)->decodeDFA(emission, dfa, dfa_size);
}

char *w2l_decoder_greedy(w2l_decoder *c_decoder, w2l_emission *emission) {
    auto decoder = reinterpret_cast<PublicDecoder *>(c_decoder);
    std::vector<int> path(emission->n_frames);
    decoder->decodeGreedy(emission, &path[0]);

    std::ostringstream ostr;
    int lastTok = -1;
    for (int tok : path) {
        if (tok == lastTok) continue;
        lastTok = tok;
        ostr << decoder->tokenDict.getEntry(tok);
    }
    auto str = ostr.str();
    return strdup(str.c_str());
}

bool w2l_make_flattrie(const char *tokens, const char *kenlm_model_path, const char *lexicon_path, const char *flattrie_path) {
    return PublicDecoder::makeFlattrie(tokens, kenlm_model_path, lexicon_path, flattrie_path);
}

} // extern "C"
