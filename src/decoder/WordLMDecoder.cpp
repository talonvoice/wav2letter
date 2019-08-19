/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "decoder/WordLMDecoder.h"

namespace w2l {

void WordLMDecoder::mergeCandidates() {
}

void WordLMDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
      const TrieNode* prevLex = prevHyp.lex;
      const int prevIdx = prevLex->idx;
      const float lexMaxScore =
          prevLex == lexicon_->getRoot() ? 0 : prevLex->maxScore;
      const LMStatePtr& prevLmState = prevHyp.lmState;

      /* (1) Try children */
      for (auto& child : prevLex->children) {
        int n = child.first;
        const TrieNodePtr& lex = child.second;
        float score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight;
        }

        // We eat-up a new token
        if (opt_.criterionType != CriterionType::CTC || prevHyp.getPrevBlank() ||
            n != prevIdx) {
          if (!lex->children.empty()) {
            candidatesAdd(
                prevLmState,
                lex.get(),
                &prevHyp,
                score + opt_.lmWeight * (lex->maxScore - lexMaxScore),
                n,
                -1,
                false // prevBlank
            );
          }
        }

        // If we got a true word
        for (int i = 0; i < lex->nLabel; i++) {
          auto lmScoreReturn = lm_->score(prevLmState, lex->label[i]);
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.lmWeight * (lmScoreReturn.second - lexMaxScore) +
                  opt_.wordScore,
              n,
              lex->label[i],
              false // prevBlank
          );
        }

        // If we got an unknown word
        if (lex->nLabel == 0 && (opt_.unkScore > kNegativeInfinity)) {
          auto lmScoreReturn = lm_->score(prevLmState, unk_);
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.lmWeight * (lmScoreReturn.second - lexMaxScore) +
                  opt_.unkScore,
              n,
              unk_,
              false // prevBlank
          );
        }
      }

      /* (2) Try same lexicon node */
      if (opt_.criterionType != CriterionType::CTC || !prevHyp.getPrevBlank()) {
        int n = prevIdx;
        float score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight;
        }

        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            -1,
            false // prevBlank
        );
      }

      /* (3) CTC only, try blank */
      if (opt_.criterionType == CriterionType::CTC) {
        int n = blank_;
        float score = prevHyp.score + emissions[t * N + n];
        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            -1,
            true // prevBlank
        );
      }
      // finish proposing
    }

    candidatesStore(hyp_[startFrame + t + 1], false);
    //// FIXME: KenLM does not need to update cache
    // updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }

  nDecodedFrames_ += T;
}

} // namespace w2l
