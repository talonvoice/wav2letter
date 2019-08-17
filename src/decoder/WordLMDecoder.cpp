/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
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

  #pragma omp parallel
  {
    for (int t = 0; t < T; t++) {
      std::deque<LexiconDecoderState> privateCandidates;
      auto prevHyps = hyp_[startFrame + t];
      #pragma omp for
      for (int i = 0; i < prevHyps.size(); i++) {
        const LexiconDecoderState& prevHyp = prevHyps[i];
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
          if (opt_.criterionType != CriterionType::CTC || prevHyp.prevBlank ||
              n != prevIdx) {
            if (!lex->children.empty()) {
              privateCandidates.emplace_back(
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
            privateCandidates.emplace_back(
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
            privateCandidates.emplace_back(
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
        if (opt_.criterionType != CriterionType::CTC || !prevHyp.prevBlank) {
          int n = prevIdx;
          float score = prevHyp.score + emissions[t * N + n];
          if (nDecodedFrames_ + t > 0 &&
              opt_.criterionType == CriterionType::ASG) {
            score += transitions_[n * N + prevIdx];
          }
          if (n == sil_) {
            score += opt_.silWeight;
          }
          privateCandidates.emplace_back(
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
          privateCandidates.emplace_back(
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

      #pragma omp single
      candidatesReset();
      #pragma omp critical
      for (auto& c : privateCandidates) {
        candidatesAdd(c);
      }
      #pragma omp single
      candidatesStore(hyp_[startFrame + t + 1], false);
      // TODO: LM Cache is unused on KenLM
      // updateLMCache(lm_, hyp_[startFrame + t + 1]);
    }
  }
  // end #pragma omp parallel

  nDecodedFrames_ += T;
}

} // namespace w2l
