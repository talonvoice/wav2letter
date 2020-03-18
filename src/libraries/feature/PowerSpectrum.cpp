/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "PowerSpectrum.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>

#include "SpeechUtils.h"

namespace w2l {

PowerSpectrum::PowerSpectrum(const FeatureParams& params)
    : featParams_(params),
      dither_(params.ditherVal),
      preEmphasis_(params.preemCoef, params.numFrameSizeSamples()),
      windowing_(params.numFrameSizeSamples(), params.windowType) {
  validatePowSpecParams();
  auto nFFt = featParams_.nFft();
  inFftBuf_.resize(nFFt, 0.0);
  outFftBuf_.resize(2 * nFFt);

  // ipps init
  m_outPerm    = ippsMalloc_64f(nFFt * 2);
  int order = log2(nFFt);
  int flags = IPP_FFT_NODIV_BY_ANY;
  IppHintAlgorithm quality = ippAlgHintNone;
  int sizeSpec, sizeInit, sizeBuffer;
  if (ippsFFTGetSize_R_64f(order, flags, quality, &sizeSpec, &sizeInit, &sizeBuffer) != ippStsNoErr) {
    throw std::runtime_error("ippsFFTGetSize_R_64f failed on order " + std::to_string(order));
  }
  m_memSpec = ippsMalloc_8u(sizeSpec);
  m_memBuffer = ippsMalloc_8u(sizeBuffer);
  Ipp8u *memInit = NULL;
  if (sizeInit > 0 ) {
    memInit = ippsMalloc_8u(sizeInit);
  }
  if (ippsFFTInit_R_64f(&m_fftSpec, order, flags, quality, m_memSpec, memInit) != ippStsNoErr) {
    throw std::runtime_error("ippsFFTInit_R_64f failed on order " + std::to_string(order));
  }
  ippFree(memInit);
}

std::vector<float> PowerSpectrum::apply(const std::vector<float>& input) {
  auto frames = frameSignal(input, featParams_);
  if (frames.empty()) {
    return {};
  }
  return powSpectrumImpl(frames);
}

std::vector<float> PowerSpectrum::powSpectrumImpl(std::vector<float>& frames) {
  int nSamples = featParams_.numFrameSizeSamples();
  int nFrames = frames.size() / nSamples;
  int nFft = featParams_.nFft();
  int K = featParams_.filterFreqResponseLen();

  if (featParams_.ditherVal != 0.0) {
    frames = dither_.apply(frames);
  }
  if (featParams_.zeroMeanFrame) {
    for (size_t f = 0; f < nFrames; ++f) {
      auto begin = frames.data() + f * nSamples;
      float mean = std::accumulate(begin, begin + nSamples, 0.0);
      mean /= nSamples;
      std::transform(
          begin, begin + nSamples, begin, [mean](float x) { return x - mean; });
    }
  }
  if (featParams_.preemCoef != 0) {
    preEmphasis_.applyInPlace(frames);
  }
  windowing_.applyInPlace(frames);
  std::vector<float> dft(K * nFrames);
  std::vector<double> dftDouble(K * nFrames);
  std::lock_guard<std::mutex> lock(fftMutex_);
  for (size_t f = 0; f < nFrames; ++f) {
      auto begin = frames.data() + f * nSamples;
      {
          std::copy(begin, begin + nSamples, inFftBuf_.data());
          ippsFFTFwd_RToPerm_64f(inFftBuf_.data(), m_outPerm, m_fftSpec, m_memBuffer);
          ippsConjPerm_64fc(m_outPerm, outFftBuf_.data(), nFft);
          ippsMagnitude_64fc(outFftBuf_.data(), &dftDouble[f * K], K);
      }
  }
  std::copy(dftDouble.begin(), dftDouble.end(), dft.data());
  return dft;
}

std::vector<float> PowerSpectrum::batchApply(
    const std::vector<float>& input,
    int batchSz) {
  if (batchSz <= 0) {
    throw std::invalid_argument("PowerSpectrum: negative batchSz");
  } else if (input.size() % batchSz != 0) {
    throw std::invalid_argument(
        "PowerSpectrum: input size is not divisible by batchSz");
  }
  int N = input.size() / batchSz;
  int outputSz = outputSize(N);
  std::vector<float> feat(outputSz * batchSz);

#pragma omp parallel for num_threads(batchSz)
  for (int b = 0; b < batchSz; ++b) {
    auto start = input.begin() + b * N;
    std::vector<float> inputBuf(start, start + N);
    auto curFeat = apply(inputBuf);
    if (outputSz != curFeat.size()) {
      throw std::logic_error("PowerSpectrum: apply() returned wrong size");
    }
    std::copy(
        curFeat.begin(), curFeat.end(), feat.begin() + b * curFeat.size());
  }
  return feat;
}

FeatureParams PowerSpectrum::getFeatureParams() const {
  return featParams_;
}

int PowerSpectrum::outputSize(int inputSz) {
  return featParams_.powSpecFeatSz() * featParams_.numFrames(inputSz);
}

void PowerSpectrum::validatePowSpecParams() const {
  if (featParams_.samplingFreq <= 0) {
    throw std::invalid_argument("PowerSpectrum: samplingFreq is negative");
  } else if (featParams_.frameSizeMs <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameSizeMs is negative");
  } else if (featParams_.frameStrideMs <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameStrideMs is negative");
  } else if (featParams_.numFrameSizeSamples() <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameSizeMs is too low");
  } else if (featParams_.numFrameStrideSamples() <= 0) {
    throw std::invalid_argument("PowerSpectrum: frameStrideMs is too low");
  }
}

PowerSpectrum::~PowerSpectrum() {
  ippFree(m_memSpec);
  ippFree(m_memBuffer);
  ippFree(m_outPerm);
}

} // namespace w2l
