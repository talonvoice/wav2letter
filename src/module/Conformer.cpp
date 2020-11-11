/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <module/Conformer.h>

using namespace fl;

namespace w2l {

fl::Variable relativePositionEmbeddingRotate(const fl::Variable& input) {
  auto data = input.array();
  int d0 = data.dims(0);
  int d1 = data.dims(1);
  int d2 = data.dims(2);
  int d3 = data.dims(3);
  data = af::join(0, data, af::constant(0.0, d1, d1, d2, d3, data.type()));
  data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  auto gradFunc = [d0, d1, d2, d3](
                      std::vector<fl::Variable>& inputs,
                      const fl::Variable& gradOutput) {
    auto gradData = gradOutput.array();
    gradData = af::moddims(gradData, af::dim4((d0 + d1 - 1) * d1, 1, d2, d3));
    gradData = af::join(
        0, gradData, af::constant(0.0, d1, 1, d2, d3, gradData.type()));
    gradData = af::moddims(gradData, af::dim4(d0 + d1, d1, d2, d3));
    gradData = gradData.rows(0, d0 - 1);
    inputs[0].addGrad(fl::Variable(gradData, false));
  };
  return fl::Variable(data, {input}, gradFunc);
}

fl::Variable multiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const fl::Variable& mask,
    const int32_t nHead,
    const double pDropout,
    const int32_t offset /* = 0 */) {
  int32_t bsz = query.dims(2);
  int32_t modelDim = query.dims(1);
  int32_t headDim = modelDim / nHead;

  auto q = moddims(query, af::dim4(-1, headDim, nHead * bsz));
  auto k = moddims(key, af::dim4(-1, headDim, nHead * bsz));
  auto v = moddims(value, af::dim4(-1, headDim, nHead * bsz));

  auto scores = matmulNT(q, k);
  if (!posEmb.isempty()) {
    int n = posEmb.dims(0) / 2 - offset;
    auto pscores = relativePositionEmbeddingRotate(matmulNT(posEmb, q));
    scores = scores + transpose(pscores.rows(n, n + k.dims(0) - 1));
  }
  scores = scores / std::sqrt(float(headDim));
  if (!mask.isempty()) {
    scores = scores + tileAs(mask, scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmul(attn, v);
  result = moddims(result, af::dim4(-1, headDim * nHead, bsz));
  return result;
}

Conformer::Conformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t bptt,
    int32_t convKernel,
    float pDropout,
    float pLayerDropout /* = 0. */)
    : nHeads_(nHeads),
      bptt_(bptt),
      convKernel_(convKernel),
      pDropout_(pDropout),
      pLayerDropout_(pLayerDropout),
      w11_(std::make_shared<Linear>(conformerInitLinear(modelDim, mlpDim))),
      w12_(std::make_shared<Linear>(conformerInitLinear(mlpDim, modelDim))),
      w21_(std::make_shared<Linear>(conformerInitLinear(modelDim, mlpDim))),
      w22_(std::make_shared<Linear>(conformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          conformerInitLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0}))),
      normMhsa_(std::make_shared<LayerNorm>(std::vector<int>({0}))),
      normConv1_(std::make_shared<LayerNorm>(std::vector<int>({2}))),
      normConv2_(std::make_shared<LayerNorm>(std::vector<int>({2}))),
      norm3_(std::make_shared<LayerNorm>(std::vector<int>({0}))),
      conv1_(std::make_shared<Conv2D>(modelDim, modelDim * 2, 1, 1)),
      conv2_(std::make_shared<Conv2D>(modelDim, modelDim, 1, 1)),
      convDepthWiseStep1_(std::make_shared<Conv2D>(
          modelDim,
          modelDim,
          convKernel,
          1,
          1,
          1,
          fl::PaddingMode::SAME,
          0,
          1,
          1,
          true,
          modelDim)),
      convDepthWiseStep2_(std::make_shared<Conv2D>(modelDim, modelDim, 1, 1)) {
  if (bptt_ > 0) {
    params_.push_back(uniform(2 * bptt - 1, headDim, -0.1, 0.1));
  }
  // first feed-forward module
  add(w11_);
  add(w12_);
  add(norm1_);
  // second feed-forward module
  add(w21_);
  add(w22_);
  add(norm2_);
  // multihead attention module
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(normMhsa_),
  // conv module
  add(conv1_);
  add(conv2_);
  add(convDepthWiseStep1_);
  add(convDepthWiseStep2_);
  add(normConv1_);
  add(normConv2_);
  // final layer norm of conformer block
  add(norm3_);
}

Conformer::Conformer() {}

Variable Conformer::conformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
}

Variable Conformer::swish(const Variable& input) {
  return input * sigmoid(input);
}

Variable Conformer::mhsa(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  int bsz = input.dims(2);

  auto normedInput = (*normMhsa_)(input);
  auto q = transpose((*wq_)(normedInput));
  auto k = transpose((*wk_)(normedInput));
  auto v = transpose((*wv_)(normedInput));

  Variable mask, posEmb;
  if (bptt_ > 0) {
    posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));
  }
  auto result = multiheadAttention(q, k, v, posEmb, mask, nHeads_, pDropout, 0);
  result = (*wf_)(transpose(result));
  result = input + dropout(result, pDropout);
  return result;
}

Variable Conformer::conv(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  // input C x T x B x 1
  auto result = reorder(input, 1, 3, 0, 2);
  // T x 1 x C x B
  // apply first pointwise conv
  result = gatedlinearunit((*conv1_)(((*normConv1_)(result))), 2);
  // apply depthwise separable convolutions
  result = (*convDepthWiseStep2_)((*convDepthWiseStep1_)(result));
  result = swish(((*normConv2_)(result)));
  // apply second pointwise conv
  result = dropout((*conv2_)(result), pDropout);
  result = reorder(result, 2, 0, 3, 1);
  // C x T x B x 1
  return result + input;
}

std::vector<Variable> Conformer::forward(const std::vector<Variable>& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerDropout_)) {
    f = 0.0;
  }
  auto x = input[0];
  // apply first feed-forward module
  auto ffn1 = x +
      dropout((*w12_)(dropout(swish((*w11_)(((*norm1_)(x)))), pDropout)),
              pDropout);
  x = x + f * 0.5 * ffn1;
  // apply multihead attention module
  x = x + f * mhsa(x);
  // apply conv module
  x = x + f * conv(x);
  // apply second feed-forward module
  auto ffn2 = x +
      dropout((*w22_)(dropout(swish((*w21_)(((*norm2_)(x)))), pDropout)),
              pDropout);
  x = x + f * 0.5 * ffn2;
  x = ((*norm3_)(x));
  return {x};
}

std::string Conformer::prettyString() const {
  std::ostringstream ss;
  ss << "Conformer "
     << "(modelDim: " << params_[1].dims(1) << "), "
     << "(mlpDim: " << params_[1].dims(0) << "), "
     << "(nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerDropout: " << pLayerDropout_ << "), "
     << "(bptt: " << bptt_ << "), "
     << "(convKernel: " << convKernel_ << ") ";
  return ss.str();
}

} // namespace w2l
