/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SpeechUtils.h"

#include <cstddef>
#include <stdexcept>
#include <chrono>
#include <iostream>

#if W2L_LIBRARIES_USE_MKL
#include <mkl_cblas.h>
#elif __APPLE__
#include <Accelerate/Accelerate.h>
#else
#define FBGEMM_CBLAS_GEMM 1
#include <fbgemm/FbgemmFP16.h>
#include <omp.h>
#endif

namespace w2l {

std::vector<float> frameSignal(
    const std::vector<float>& input,
    const FeatureParams& params) {
  auto frameSize = params.numFrameSizeSamples();
  auto frameStride = params.numFrameStrideSamples();
  int numframes = params.numFrames(input.size());
  // HTK: Values coming out of rasta treat samples as integers,
  // not range -1..1, hence scale up here to match (approx)
  float scale = 32768.0;
  std::vector<float> frames(numframes * frameSize);
  for (size_t f = 0; f < numframes; ++f) {
    for (size_t i = 0; i < frameSize; ++i) {
      frames[f * frameSize + i] = scale * input[f * frameStride + i];
    }
  }
  return frames;
}

#ifdef FBGEMM_CBLAS_GEMM
std::vector<float> cblasGemm(
    const std::vector<float>& matA,
    const std::vector<float>& matB,
    int n,
    int k) {
  if (n <= 0 || k <= 0 || matA.empty() || (matA.size() % k != 0) ||
      (matB.size() != n * k)) {
    throw std::invalid_argument("cblasGemm: invalid arguments");
  }

  int m = matA.size() / k;

  std::vector<float> matC(m * n);

  fbgemm::PackedGemmMatrixFP16 Bp(fbgemm::matrix_op_t::NoTranspose, k, n, 1.0, matB.data());
#pragma omp parallel
    {
      int num_threads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      cblas_gemm_compute(fbgemm::matrix_op_t::NoTranspose, m, matA.data(), Bp, 0.0, matC.data(), tid, num_threads);
    }

  return matC;
};
#else // FBGEMM_CBLAS_GEMM

std::vector<float> cblasGemm(
    const std::vector<float>& matA,
    const std::vector<float>& matB,
    int n,
    int k) {
  if (n <= 0 || k <= 0 || matA.empty() || (matA.size() % k != 0) ||
      (matB.size() != n * k)) {
    throw std::invalid_argument("cblasGemm: invalid arguments");
  }
  int m = matA.size() / k;

  std::vector<float> matC(m * n);
  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      m,
      n,
      k,
      1.0, // alpha
      matA.data(),
      k,
      matB.data(),
      n,
      0.0, // beta
      matC.data(),
      n);
  return matC;
};

#endif // FBGEMM_CBLAS_GEMM

} // namespace w2l
