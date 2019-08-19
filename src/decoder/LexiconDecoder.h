/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <queue>

#include "decoder/Decoder.h"
#include "decoder/LM.h"
#include "decoder/Trie.h"

namespace w2l {
/**
 * LexiconDecoderState stores information for each hypothesis in the beam.
 */

#pragma pack(push, 1)

struct LexiconDecoderState {
  LMStatePtr lmState; // Language model state
  const TrieNode* lex; // Trie node in the lexicon
  const LexiconDecoderState* parent; // Parent hypothesis
  /* tag represents bitwise:
   * int word : 23
   * bool prevBlank : 1
   * int token : 8
   */
  uint32_t tag;
  float score; // Score so far

  LexiconDecoderState(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const float score,
      const int token,
      const int word,
      const bool prevBlank = false)
      : lmState(lmState),
        lex(lex),
        parent(parent),
        score(score) {
          setToken(token);
          setWord(word);
          setPrevBlank(prevBlank);
        }

  LexiconDecoderState()
      : lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
        score(0),
        tag(0xfffffeff) {}

  int getToken() const {
    int16_t token = tag & 0xFF;
    return token;
  }
  void setToken(int token) {
    tag = (tag & ~0xFF) | (token & 0xFF);
  }

  int getWord() const {
    int32_t word = (tag & 0xFFFFFE00);
    return word >> 9;
  }
  void setWord(int word) {
    tag = (tag & ~0xFFFFFE00) | ((word << 9) & 0xFFFFFE00);
  }

  bool getPrevBlank() const {
    return (tag >> 8) & 1;
  }
  void setPrevBlank(bool prevBlank) {
    tag = (tag & ~(1 << 8)) | (prevBlank & 1) << 8;
  }

  bool isComplete() const {
    return !parent || parent->getWord() != -1;
  }

  bool operator<(const LexiconDecoderState &other) const {
      return score < other.score;
  }
};
#pragma pack(pop)

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known| + unkScore_ *
 * |W_unknown| - silWeight_ * |{i| pi_i = <sil>}|
 *
 * where P_{lm}(W) is the language model score, pi_i is the value for the i-th
 * frame in the path leading to W and AM(W) is the (unnormalized) acoustic model
 * score of the transcription W. Note that the lexicon is used to limit the
 * search space and all candidate words are generated from it if unkScore is
 * -inf, otherwise <UNK> will be generated for OOVs.
 */
class LexiconDecoder : public Decoder {
 public:
  LexiconDecoder(
      const DecoderOptions& opt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int sil,
      const int blank,
      const int unk,
      const std::vector<float>& transitions)
      : Decoder(opt),
        lexicon_(lexicon),
        lm_(lm),
        transitions_(transitions),
        sil_(sil),
        blank_(blank),
        unk_(unk) {}

  void decodeBegin() override;

  void decodeEnd() override;

  void dumpBeams();

  int nHypothesis() const;

  int nDecodedFramesInBuffer() const;

  void prune(int lookBack = 0) override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  TriePtr lexicon_;
  LMPtr lm_;
  std::vector<float> transitions_;

  // All the hypothesis new candidates (can be larger than beamsize) proposed
  // based on the ones from previous frame
  std::priority_queue<LexiconDecoderState> candidates_;

  // Best candidate score of current frame
  float candidatesBestScore_;

  // Index of silence label
  int sil_;

  // Index of blank label (for CTC)
  int blank_;

  // Index of unknown word
  int unk_;

  // Vector of hypothesis for all the frames so far
  std::unordered_map<int, std::vector<LexiconDecoderState>> hyp_;

  // These 2 variables are used for online decoding, for hypothesis pruning
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.

  // Reset candidates buffer for decoding a new input frame
  void candidatesReset();

  // Add a new candidate to the buffer
  void candidatesAdd(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const float score,
      const int token,
      const int label,
      const bool prevBlank);

  // Merge and sort candidates proposed in the current frame and place them into
  // the `hyp_` buffer
  void candidatesStore(
      std::vector<LexiconDecoderState>& nextHyp,
      const bool isSort);

  // Merge hypothesis getting into same state from different path
  virtual void mergeCandidates() = 0;
};

} // namespace w2l
