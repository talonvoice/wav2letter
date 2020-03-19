typedef struct w2lstream w2lstream;

w2lstream *w2lstream_new(const char *feature_model_path, const char *acoustic_model_path, const char *tokens_path, int chunk_size);
char *w2lstream_run(w2lstream *engine, int64_t stream_id, float *samples, size_t sample_count);
void w2lstream_reset(w2lstream *engine, int64_t stream_id);
void w2lstream_free(w2lstream *engine);
