#include "w2l.h"
#include "w2l_encode.h"
#include "w2l_encode_backend.h"

extern "C" {

w2l_engine *w2l_engine_new() {
    // TODO: what other engine config do I need?
    auto engine = new Engine();
    return reinterpret_cast<w2l_engine *>(engine);
}

bool w2l_engine_load_w2l(w2l_engine *engine, const char *acoustic_model_path, const char *tokens_path) {
    return reinterpret_cast<Engine *>(engine)->loadW2lModel(acoustic_model_path, tokens_path);
}

bool w2l_engine_load_b2l(w2l_engine *engine, const char *path) {
    return reinterpret_cast<Engine *>(engine)->loadB2lModel(path);
}

bool w2l_engine_export_w2l(w2l_engine *engine, const char *path) {
    return reinterpret_cast<Engine *>(engine)->exportW2lModel(path);
}

bool w2l_engine_export_b2l(w2l_engine *engine, const char *path) {
    return reinterpret_cast<Engine *>(engine)->exportB2lModel(path);
}

w2l_emission *w2l_engine_forward(w2l_engine *engine, float *samples, size_t sample_count) {
    return reinterpret_cast<Engine *>(engine)->forward(samples, sample_count);
}

void w2l_engine_free(w2l_engine *engine) {
    if (engine) {
        delete reinterpret_cast<Engine *>(engine);
    }
}

} // extern "C"
