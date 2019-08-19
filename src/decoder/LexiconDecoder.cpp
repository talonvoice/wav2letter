/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <float.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "decoder/LexiconDecoder.h"

namespace w2l {

void LexiconDecoder::candidatesReset() {
  candidatesBestScore_ = kNegativeInfinity;
  candidates_ = std::priority_queue<LexiconDecoderState>();
}

void LexiconDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const TrieNode* lex,
    const LexiconDecoderState* parent,
    const float score,
    const int token,
    const int word,
    const bool prevBlank) {
  if (isValidCandidate(candidatesBestScore_, score, opt_.beamThreshold)) {
    candidates_.emplace(
        lmState, lex, parent, score, token, word, prevBlank);
  }
}

// used for making an unordered_set of LexiconDecoderStates based on their LMStatePtr
struct StateHash {
  LMPtr lm_;
  size_t operator()(const LexiconDecoderState *v) const {
    return lm_->stateHash(v->lmState) ^ size_t(v->lex);
  }
};

struct StateEquality {
  LMPtr lm_;
  int operator()(const LexiconDecoderState *v1, const LexiconDecoderState *v2) const {
    return v1->lex == v2->lex && lm_->compareState(v1->lmState, v2->lmState) == 0;
  }
};

void LexiconDecoder::candidatesStore(
    std::vector<LexiconDecoderState>& nextHyp,
    const bool returnSorted) {
  nextHyp.clear();
  nextHyp.reserve(std::min<size_t>(candidates_.size(), opt_.beamSize));

  std::unordered_set<const LexiconDecoderState *, StateHash, StateEquality>
    seen(opt_.beamSize * 2, StateHash{lm_}, StateEquality{lm_});;

  while (nextHyp.size() < opt_.beamSize && !candidates_.empty()) {
    auto& c = std::move(candidates_.top());
    if (c.score < candidatesBestScore_ - opt_.beamThreshold) {
      break;
    }
    if (seen.find(&c) == seen.end()) {
      nextHyp.emplace_back(std::move(c));
      seen.emplace(&nextHyp.back());
    }
    candidates_.pop();
  }
}

void LexiconDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconDecoderState>());

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(
      lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, sil_, -1);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

static std::string dot_header(R"(digraph beam {
  rankdir=LR
  splines=false
  node[shape=box, style=rounded]
  edge[arrowhead=none, headport=w, tailport=e]
)");
static std::string dot_footer("}");
static std::string token_lookup("|'abcdefghijklmnopqrstuvwxyz");

void LexiconDecoder::dumpBeams() {
  //// write dotfile of all decoder beams
  // dot -Tpng -o graph.png graph.dot

  // pick first free graph-%d.dot filename and open for writing
  int64_t i = 0;
  std::string name = "";
  while (true) {
    name = "graph-" + std::to_string(i++) + ".dot";
    std::ifstream f(name);
    if (!f.good()) {
      break;
    }
  }
  std::ofstream outfile(name);
  std::cout << "writing: " << name << std::endl;

  outfile << dot_header;

  // nodes are named N0_0
  // as in, Nframe_beamindex
  // where N0_0 is the best beam of timestep 0

  // this set tracks linked nodes, with outbound edges, which makes them not "dead ends"
  // we need to know about dead ends so we can color them red
  std::unordered_set<const LexiconDecoderState *> linked;

  std::string indent = "  ";
  // write horizontal edges
  for (int t = 1; t < hyp_.size(); t++) {
    const auto& prevFrame = hyp_[t-1];
    const auto& frame = hyp_[t];

    std::unordered_map<const LexiconDecoderState *, int> stateToIndex;
    for (int i = 0; i < prevFrame.size(); i++) {
      stateToIndex[&prevFrame[i]] = i;
    }
    for (int beam = 0; beam < frame.size(); beam++) {
      linked.insert(frame[beam].parent);
      int parent = stateToIndex[frame[beam].parent];
      outfile << indent
        << "N" << t-1 << "_" << parent << " -> "
        << "N" << t   << "_" << beam   << "\n";
    }
  }
  outfile << "\n";

  // write labels
  for (int t = 0; t < hyp_.size(); t++) {
    const auto& frame = hyp_[t];
    for (int beam = 0; beam < frame.size(); beam++) {
      const auto& state = frame[beam];
      std::string label = token_lookup.substr(state.getToken(), 1);
      outfile << indent <<
        "N" << t << "_" << beam << "[label=\"" << label << "\"";

      // mark dead ends as read
      if (linked.find(&state) == linked.end()) {
        outfile << ", color=\"red\"";
      }
      outfile << "]\n";
    }
  }

  // write vertical ranks
  for (int t = 0; t < hyp_.size(); t++) {
    const auto& prevHyp = hyp_[t];
    if (prevHyp.size() > 0) {
      outfile << indent << "{ rank = same; ";
      for (int beam = 0; beam < prevHyp.size(); beam++) {
        outfile << "N" << t << "_" << beam << "; ";
      }
      outfile << "}\n";
    }
  }
  outfile << "\n";

  // write horizontal/vertical matrix edges
  outfile << indent << "node[style=invis]\n";
  outfile << indent << "edge[style=invis]\n";
  // vertical matrix edges
  for (int t = 0; t < hyp_.size(); t++) {
    const auto& prevHyp = hyp_[t];
    if (prevHyp.size() > 0) {
      outfile << indent << "N" << t << "_" << 0;
      for (int beam = 1; beam < prevHyp.size(); beam++) {
        outfile << " -> " << "N" << t << "_" << beam;
      }
      outfile << "\n";
    }
  }
  // horizontal matrix edges
  for (int beam = 0; beam < opt_.beamSize; beam++) {
    outfile << indent << "N" << 0 << "_" << beam;
    for (int t = 1; t < hyp_.size(); t++) {
      outfile << " -> " << "N" << t << "_" << beam;
    }
  }
  outfile << "\n";

  // write footer
  outfile << dot_footer;
  //// end dotfile writer
}

void LexiconDecoder::decodeEnd() {
  dumpBeams();
  candidatesReset();
  for (const LexiconDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    const TrieNode* prevLex = prevHyp.lex;
    const LMStatePtr& prevLmState = prevHyp.lmState;

    auto lmStateScorePair = lm_->finish(prevLmState);
    candidatesAdd(
        lmStateScorePair.first,
        prevLex,
        &prevHyp,
        prevHyp.score + opt_.lmWeight * lmStateScorePair.second,
        -1,
        -1,
        false // prevBlank
    );
  }

  candidatesStore(hyp_[nDecodedFrames_ - nPrunedFrames_ + 1], true);
  ++nDecodedFrames_;
}

std::vector<DecodeResult> LexiconDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  if (finalFrame < 1) {
    return std::vector<DecodeResult>{};
  }

  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconDecoder::getBestHypothesis(int lookBack) const {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return DecodeResult();
  }

  const LexiconDecoderState* bestNode = findBestAncestor(
      hyp_.find(nDecodedFrames_ - nPrunedFrames_)->second, lookBack);
  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  const LexiconDecoderState* bestNode = findBestAncestor(
      hyp_.find(nDecodedFrames_ - nPrunedFrames_)->second, lookBack);
  if (!bestNode) {
    return; // Not enough decoded frames to prune
  }

  int startFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  if (startFrame < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (2) Move things from back of hyp_ to front and normalize scores */
  pruneAndNormalize(hyp_, startFrame, lookBack);

  nPrunedFrames_ = nDecodedFrames_ - lookBack;
}

} // namespace w2l
