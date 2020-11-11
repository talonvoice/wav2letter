/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

/**
 * A module which implements a Conformer block.
 *
 * For details, see [Gulati et al
 * (2020)](https://arxiv.org/pdf/2005.08100.pdf).
 *
 * Input dimension at forward is assumed to be CxTxBx1, where C is the
 * number of features, T the sequence length and B the batch size.
 * @param modelDim input embedding dimension
 * @param headDim dimension of each head
 * @param mlpDim dimension of the feed-forward layers
 * @param nHeads number of heads
 * @param bptt size for learnt relative positional
 * embedding matrix (2 * bptt - 1) * nHeads
 * @param convKernel convolution layers kernel
 * @param pDropout dropout probability
 * @param pLayerdrop layer dropout probability
 */
class Conformer : public fl::Container {
 public:
  explicit Conformer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t bptt,
      int32_t convKernel,
      float pDropout,
      float pLayerDropout = 0.);

  std::vector<fl::Variable> forward(const std::vector<fl::Variable>& input) override;
  std::string prettyString() const override;

 private:
  int32_t nHeads_;
  int32_t bptt_;
  int32_t convKernel_;
  double pDropout_;
  float pLayerDropout_;

  std::shared_ptr<fl::Linear> w11_, w12_, w21_, w22_, wq_, wk_, wv_, wf_;
  std::shared_ptr<fl::LayerNorm> norm1_, norm2_, normMhsa_, normConv1_, normConv2_,
      norm3_;
  std::shared_ptr<fl::Conv2D> conv1_, conv2_, convDepthWiseStep1_,
      convDepthWiseStep2_;

  static fl::Variable conformerInitLinear(int32_t inDim, int32_t outDim);
  fl::Variable swish(const fl::Variable& input);
  fl::Variable mhsa(const fl::Variable& input);
  fl::Variable conv(const fl::Variable& input);

  Conformer();

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      w11_,
      w12_,
      w21_,
      w22_,
      wq_,
      wk_,
      wv_,
      wf_,
      normMhsa_,
      norm1_,
      norm2_,
      norm3_,
      normConv1_,
      normConv2_,
      conv1_,
      conv2_,
      convDepthWiseStep1_,
      convDepthWiseStep2_,
      nHeads_,
      pDropout_,
      pLayerDropout_,
      bptt_,
      convKernel_)
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::Conformer)
