#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// malloc(sizeof(w2l_emission) + sizeof(float) * n_nrames * n_tokens);
typedef struct __attribute__((packed)) w2l_emission {
    int32_t n_frames;
    int32_t n_tokens;
    float matrix[0];
} w2l_emission;

#ifdef __cplusplus
} // extern "C"
#endif
