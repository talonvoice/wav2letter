typedef struct w2lstream w2lstream;

w2lstream *w2lstream_new(const char *feature_model_path, const char *acoustic_model_path, const char *tokens_path, int chunk_size);
char *w2lstream_run(w2lstream *engine, float *samples, size_t sample_count);
void w2lstream_free(w2lstream *engine);
