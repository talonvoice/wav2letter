Dictionary *globalTokens;

using namespace w2l;

namespace KenFlatTrieLM {
    // A single LM reference gets passed around. It's for context data.
    struct LM {
        KenLMPtr ken;
        FlatTriePtr trie;
    };

    // Every DecoderState will store a State.
    struct State {
        LMStatePtr kenState;
        const FlatTrieNode *lex = nullptr;

        // used for making an unordered_set of const State*
        struct Hash {
            const LM &lm_;
            size_t operator()(const State *v) const {
                return lm::ngram::hash_value(*static_cast<KenLMState*>(v->kenState.get())->ken()) ^ size_t(v->lex);
            }
        };

        struct Equality {
            const LM &lm_;
            int operator()(const State *v1, const State *v2) const {
                return v1->lex == v2->lex && v1->kenState == v2->kenState;
            }
        };

        // For avoiding early shared_ptr copies; kenState should have better lifetime management!
        // The problem is that forChildren needs to create new States, but often these states
        // get rejected immediately because of insufficient score. So forChildren uses Proxys
        // instead which only get actualize()d when necessary.
        struct Proxy {
            const State &base;
            const FlatTrieNode *lex = nullptr;

            float maxWordScore() const {
                return lex->maxScore;
            }

            // Iterate over labels, calling fn with: the new State, the label index and the lm score
            template <typename Fn>
            void forLabels(const LM &lm, Fn&& fn) const {
                const auto n = lex->nLabel;
                for (int i = 0; i < n; ++i) {
                    int label = lex->label(i);
                    auto kenAndScore = lm.ken->score(base.kenState, label);
                    State it;
                    it.kenState = std::move(kenAndScore.first);
                    it.lex = lm.trie->getRoot();
                    fn(std::move(it), label, kenAndScore.second);
                }
            }

            State actualize() const {
                return State{base.kenState, lex};
            }
        };

        // Call finish() on the lm, like for end-of-sentence scoring
        std::pair<State, float> finish(const LM &lm) const {
            State result = *this;
            auto p = lm.ken->finish(kenState);
            result.kenState = p.first;
            return {result, p.second};
        }

        float maxWordScore() const {
            return lex->maxScore;
        }

        // Iterate over children of the state, calling fn with:
        // new State (or a Proxy), new token index and whether the new state has children
        template <typename Fn>
        bool forChildren(int frame, std::unordered_set<int> &indices, const LM &lm, Fn&& fn) const {
            const auto n = lex->nChildren;
            for (int i = 0; i < n; ++i) {
                auto nlex = lex->child(i);
                if (indices.find(nlex->idx) != indices.end()) {
                    fn(Proxy{*this, nlex}, nlex->idx, nlex->nChildren > 0);
                }
            }
            return true;
        }

        State &actualize() {
            return *this;
        }
    };
};

#pragma pack(push, 1)
template <typename LMStateType>
struct SimpleDecoderState {
  LMStateType lmState;
  const SimpleDecoderState* parent; // Parent hypothesis
  /* tag represents bitwise:
   * int word : 23
   * bool prevBlank : 1
   * int token : 8
   */
  uint32_t tag;
  float score; // Score so far
  int16_t token;

  SimpleDecoderState(
      LMStateType lmState,
      const SimpleDecoderState* parent,
      const float score,
      const int token,
      const int word,
      const bool prevBlank = false)
      : lmState(std::move(lmState)),
        parent(parent),
        score(score) {
          setToken(token);
          setWord(word);
          setPrevBlank(prevBlank);
        }

  SimpleDecoderState()
      : parent(nullptr),
        score(0),
        tag(0xfffffeff) {}

  int getToken() const {
    return this->token;
  }
  void setToken(int token) {
      this->token = token;
  }

  int getWord() const {
    int32_t word = (tag & 0xFFFFFE00);
    if (word == 0xFFFFFE00)
        return -1;
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

  bool operator<(const SimpleDecoderState &other) const {
      return score < other.score;
  }

  bool operator>(const SimpleDecoderState &other) const {
      return score > other.score;
  }
};
#pragma pack(pop)

template <typename LMStateType1, typename LMStateType2>
static void beamSearchNewCandidate(
        std::vector<SimpleDecoderState<LMStateType2>> &candidates,
        float &bestScore,
        const DecoderOptions &opt,
        LMStateType1 lmState,
        const SimpleDecoderState<LMStateType2>* parent,
        const float score,
        const int token,
        const int word,
        const bool prevBlank = false)
{
    if (score < bestScore - opt.beamThreshold)
        return;
    if (candidates.size() >= opt.beamSize * 2 && candidates[0].score > score) {
        return;
    }
    bestScore = std::max(bestScore, score);
    candidates.emplace_back(
            std::move(lmState.actualize()), parent, score, token, word, prevBlank);
    if (candidates.size() == opt.beamSize * 2) {
        // saves us a push + pop
        std::pop_heap(candidates.begin(), candidates.end(), std::greater<SimpleDecoderState<LMStateType2>>());
        candidates.pop_back();
    } else {
        std::push_heap(candidates.begin(), candidates.end(), std::greater<SimpleDecoderState<LMStateType2>>());
    }
}

// Take at most beamSize items from candidates and fill nextHyp.
template <typename LMStateType, typename LM>
static void beamSearchSelectBestCandidates(
        std::vector<SimpleDecoderState<LMStateType>>& nextHyp,
        std::vector<SimpleDecoderState<LMStateType>>& candidates,
        const float scoreThreshold,
        const LM &lm,
        const int beamSize)
{
    nextHyp.clear();
    nextHyp.reserve(std::min<size_t>(candidates.size(), beamSize));

    std::unordered_set<const LMStateType *, typename LMStateType::Hash, typename LMStateType::Equality>
            seen(beamSize * 2, typename LMStateType::Hash{lm}, typename LMStateType::Equality{lm});;

    std::make_heap(candidates.begin(), candidates.end());

    while (nextHyp.size() < beamSize && candidates.size() > 0) {
        auto& c = candidates[0];
        if (c.score < scoreThreshold) {
            break;
        }
        auto it = seen.find(&c.lmState);
        if (it == seen.end()) {
            nextHyp.emplace_back(std::move(c));
            seen.emplace(&nextHyp.back().lmState);
        }
        std::pop_heap(candidates.begin(), candidates.end());
        candidates.resize(candidates.size() - 1);
    }
}

// Wrap up by calling lmState->finish for all hyps
template <typename LMStateType, typename LM>
static void beamSearchFinish(
        std::vector<SimpleDecoderState<LMStateType>> &hypOut,
        std::vector<SimpleDecoderState<LMStateType>> &hypIn,
        const LM &lm,
        const DecoderOptions &opt)
{
    float candidatesBestScore = -INFINITY;
    std::vector<SimpleDecoderState<LMStateType>> candidates;
    candidates.reserve(hypIn.size());

    for (const auto& prevHyp : hypIn) {
        const auto& prevLmState = prevHyp.lmState;

        auto lmStateScorePair = prevLmState.finish(lm);
        beamSearchNewCandidate(
                    candidates,
                    candidatesBestScore,
                    opt,
                    lmStateScorePair.first,
                    &prevHyp,
                    prevHyp.score + opt.lmWeight * lmStateScorePair.second,
                    -1,
                    -1
                    );
    }

    beamSearchSelectBestCandidates(hypOut, candidates,
                                   candidatesBestScore - opt.beamThreshold, lm, opt.beamSize);
}

// I want something like a strided iterator later, and it's just a hassle,
// so I make my own range abstraction :/
template <typename T>
auto rangeAdapter(const std::vector<T> &v, int start = 0, int stride = 1)
{
    size_t i = start;
    size_t size = v.size();
    return [&v, i, size, stride]() mutable -> const T* {
        if (i >= size)
            return nullptr;
        auto res = &v[i];
        i += stride;
        return res;
    };
}

// wip idea: can customize beamsearch without performance penalty by
// providing struct confirming to this interface
template <typename DecoderState>
struct DefaultHooks
{
    float extraNewTokenScore(int frame, const DecoderState &prevState, int token)
    {
        return 0;
    }

    static DefaultHooks instance;
};

template <typename DecoderState>
DefaultHooks<DecoderState> DefaultHooks<DecoderState>::instance;

template <typename LM, typename LMStateType>
struct BeamSearch
{
    using DecoderState = SimpleDecoderState<LMStateType>;

    DecoderOptions opt_;
    const std::vector<float> &transitions_;
    const LM &lm_;
    int sil_;
    int blank_;
    int unk_;
    int nTokens_;

    struct Result
    {
        std::vector<std::vector<DecoderState>> hyp;
        float bestBeamScore;
    };

    template <typename Range, typename Hooks = DefaultHooks<DecoderState>>
    Result run(w2l_emission *emission,
               const int startFrame,
               const int frames,
               Range initialHyp,
               Hooks &hooks = DefaultHooks<DecoderState>::instance) const;
};

template <typename LM, typename LMStateType>
template <typename Range, typename Hooks>
auto BeamSearch<LM, LMStateType>::run(
        w2l_emission *emission,
        const int startFrame,
        const int frames,
        Range initialHyp,
        Hooks &hooks) const
    -> Result
{
    float *emissions = &emission->matrix[0];
    std::vector<std::vector<DecoderState>> hyp;
    hyp.resize(frames + 1, std::vector<DecoderState>());

    std::vector<DecoderState> ends;
    float endsBestScore = -INFINITY;

    std::vector<DecoderState> candidates;
    candidates.reserve(opt_.beamSize);
    float candidatesBestScore = kNegativeInfinity;

    std::vector<int> indices(nTokens_);
    std::iota(indices.begin(), indices.end(), 0);
    std::unordered_set<int> indexSet(indices.begin(), indices.end());
    for (int t = 0; t < frames; t++) {
        // std::cout << "\nframe: " << t << "\n";
        int frame = startFrame + t;
        if (nTokens_ > opt_.beamSizeToken) {
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(
                indices.begin(),
                indices.begin() + opt_.beamSizeToken,
                indices.end(),
                [&](const int& left, const int& right) {
                    return emissions[frame * nTokens_ + left] > emissions[frame * nTokens_ + right];
                });
            indexSet.clear();
            indexSet.insert(indices.begin(), indices.begin() + opt_.beamSizeToken);
        }
        candidates.clear();

        float maxEmissionScore = -INFINITY;
        int maxEmissionToken = -1;
        for (int i = 0; i < nTokens_; ++i) {
            float e = emissions[frame * nTokens_ + i];
            if (e > maxEmissionScore) {
                maxEmissionScore = e;
                maxEmissionToken = i;
            }
        }
        //std::cout << t << " " << token_lookup[maxToken] << ": " << maxWeight << ", sil: " << silEm << std::endl;

        candidatesBestScore = kNegativeInfinity;

        int IDX = 0;
        auto range = t == 0 ? initialHyp : rangeAdapter(hyp[t]);
        while (auto hypIt = range()) {
            const auto& prevHyp = *hypIt;
            const auto& prevLmState = prevHyp.lmState;
            const int prevIdx = prevHyp.getToken();
            bool repeatPrevLex = true;

            /*
            const DecoderState *node = &prevHyp;
            std::cout << IDX++ << ": " << prevHyp.score << " ";
            std::vector<std::string> toks;
            while (node) {
                toks.push_back(globalTokens->getEntry(node->getToken()));
                node = node->parent;
            }
            for (int i = toks.size() - 1; i >= 0; i--) {
                std::cout << toks[i];
            }
            std::cout << "\n";
            */

            const float prevMaxScore = prevLmState.maxWordScore();
            /* (1) Try children */
            repeatPrevLex &= prevLmState.forChildren(t, indexSet, lm_, [&, prevIdx, prevMaxScore](auto lmState, int n, bool hasChildren) {
                if (n == prevIdx && (opt_.criterionType != CriterionType::CTC || !prevHyp.getPrevBlank()))
                    repeatPrevLex = true;
                float score = prevHyp.score + emissions[frame * nTokens_ + n];
                if (frame > 0 && transitions_.size() > 0) {
                    score += transitions_[n * nTokens_ + prevIdx];
                }
                if (n == sil_ || n == blank_) {
                    score += opt_.silScore;
                }
                score += hooks.extraNewTokenScore(frame, prevHyp, n);

                // If we got a true word
                bool hadLabel = false;
                lmState.forLabels(lm_, [&, prevMaxScore, score](LMStateType labelLmState, int label, float lmScore) {
                    hadLabel = true;
                    float lScore = score + opt_.lmWeight * (lmScore - prevMaxScore) + opt_.wordScore;
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_,
                            std::move(labelLmState),
                            &prevHyp,
                            lScore,
                            n,
                            label
                            );
                });

                // We eat-up a new token
                if (hasChildren && (opt_.criterionType != CriterionType::CTC || prevHyp.getPrevBlank() || n != prevIdx)) {
                    float lScore = score + opt_.lmWeight * (lmState.maxWordScore() - prevMaxScore);
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_,
                            std::move(lmState),
                            &prevHyp,
                            lScore,
                            n,
                            -1
                            );
                }

                // If we got an unknown word
                if (!hadLabel && (opt_.unkScore > kNegativeInfinity)) {
                    /*
                    auto lmScoreReturn = lm_.score(prevLmState, unk_);
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_,
                            lmScoreReturn.first,
                            &prevHyp,
                            score + opt_.lmWeight * (lmScoreReturn.second - prevMaxScore) + opt_.unkScore,
                            n,
                            unk_
                            );
                    */
                }
            });

            /* Try same lexicon node */
            if (repeatPrevLex) {
                int n = prevIdx;
                float score = prevHyp.score + emissions[frame * nTokens_ + n];
                if (frame > 0 && transitions_.size() > 0) {
                    score += transitions_[n * nTokens_ + prevIdx];
                }
                if (n == sil_ || n == blank_) {
                    score += opt_.silScore;
                }
                score += hooks.extraNewTokenScore(frame, prevHyp, n);

                beamSearchNewCandidate(
                        candidates,
                        candidatesBestScore,
                        opt_,
                        prevLmState,
                        &prevHyp,
                        score,
                        n,
                        -1
                        );
            }

            /* CTC only, try blank */
            if (opt_.criterionType == CriterionType::CTC) {
                int n = blank_;
                double score = prevHyp.score + emissions[frame * nTokens_ + n] + opt_.silScore;
                beamSearchNewCandidate(
                        candidates,
                        candidatesBestScore,
                        opt_,
                        prevLmState,
                        &prevHyp,
                        score,
                        n,
                        -1,
                        true // prevBlank
                        );
            }
        }
        beamSearchSelectBestCandidates(hyp[t + 1], candidates, candidatesBestScore - opt_.beamThreshold, lm_, opt_.beamSize);
    }

    // std::cout << "\n";
    return Result{std::move(hyp), candidatesBestScore};
}

template <typename LM, typename LMStateType>
struct SimpleDecoder
{
    using DecoderState = SimpleDecoderState<LMStateType>;
    using Search = BeamSearch<LM, LMStateType>;

    LM lm_;
    int sil_;
    int blank_;
    int unk_;
    std::vector<float> transitions_;

    DecodeResult normal(DecoderOptions &opt,
                        w2l_emission *emission,
                        const LMStateType startState) const;

    // Like normal, but returns the full beam search state
    // Also, doesn't finish() the language model
    template <typename BeamHooks = DefaultHooks<DecoderState>>
    auto normalAll(DecoderOptions &opt,
                   w2l_emission *emission,
                   const std::vector<DecoderState> &startStates,
                   BeamHooks &hooks = DefaultHooks<DecoderState>::instance) const
        -> typename Search::Result;

    template <typename BeamHooks>
    auto groupThreading(DecoderOptions &opt,
                        w2l_emission *emission,
                        const std::vector<DecoderState> &startStates,
                        BeamHooks &hooks,
                        const int nThreads,
                        const int stepsPerFanout,
                        const int threadBeamSize) const
        -> typename Search::Result;

//    DecodeResult diversity(DecoderOptions &opt,
//                           const float *emissions,
//                           const int frames,
//                           const int nTokens) const;
};

// taken from WordUtils
template <class DecoderState>
static DecodeResult _getHypothesis(const DecoderState* node, const int finalFrame) {
  const DecoderState* node_ = node;
  if (!node_) {
    return DecodeResult();
  }

  DecodeResult res(finalFrame + 1);
  res.score = node_->score;

  int i = 0;
  while (node_) {
    res.words[finalFrame - i] = node_->getWord();
    res.tokens[finalFrame - i] = node_->getToken();
    node_ = node_->parent;
    i++;
  }

  return res;
}

template <typename LM, typename LMStateType>
auto SimpleDecoder<LM, LMStateType>::normal(
        DecoderOptions &opt,
        w2l_emission *emission,
        const LMStateType startState) const
    -> DecodeResult
{
    std::vector<std::vector<DecoderState>> hyp;
    hyp.resize(1);
    hyp[0].emplace_back(startState, nullptr, 0.0, sil_, -1);

    Search beamSearch{
        .opt_ = opt,
        .transitions_ = transitions_,
        .lm_ = lm_,
        .sil_ = sil_,
        .blank_ = blank_,
        .unk_ = unk_,
        .nTokens_ = emission->n_tokens,
    };

    auto beams = beamSearch.run(emission, 0, emission->n_frames, rangeAdapter(hyp[0]));

    std::vector<DecoderState> result;
    beamSearchFinish(result, beams.hyp.back(), lm_, opt);

    return _getHypothesis(&result[0], beams.hyp.size());
}

template <typename LM, typename LMStateType>
template <typename BeamHooks>
auto SimpleDecoder<LM, LMStateType>::normalAll(
        DecoderOptions &opt,
        w2l_emission *emission,
        const std::vector<DecoderState> &startStates,
        BeamHooks &hooks) const
    -> typename Search::Result
{
    Search beamSearch{
        .opt_ = opt,
        .transitions_ = transitions_,
        .lm_ = lm_,
        .sil_ = sil_,
        .blank_ = blank_,
        .unk_ = unk_,
        .nTokens_ = emission->n_tokens,
    };

    return beamSearch.run(emission, 0, emission->n_frames, rangeAdapter(startStates), hooks);
}

template <typename LM, typename LMStateType>
template <typename BeamHooks>
auto SimpleDecoder<LM, LMStateType>::groupThreading(
        DecoderOptions &opt,
        w2l_emission *emission,
        const std::vector<DecoderState> &startStates,
        BeamHooks &hooks,
        const int nThreads,
        const int stepsPerFanout,
        const int threadBeamSize) const
    -> typename Search::Result
{
    // Run Q-steps of beamseach on subgroups of hyp
    int Q = stepsPerFanout;
    int n_groups = nThreads;
    int frames = emission->n_frames;

    typename Search::Result result;
    // essential to avoid reallocations
    result.hyp.reserve((frames + 2) * n_groups);
    result.hyp.push_back(startStates);

    std::vector<DecoderState> candidates;
    float candidatesBestScore = -INFINITY;
    std::vector<DecoderState> *startHyp = &result.hyp.back();

    #pragma omp parallel num_threads(nThreads)
    {
        int t = 0;

        Search beamSearch{
            .opt_ = opt,
            .transitions_ = transitions_,
            .lm_ = lm_,
            .sil_ = sil_,
            .blank_ = blank_,
            .unk_ = unk_,
            .nTokens_ = emission->n_tokens,
        };
        beamSearch.opt_.beamSize = threadBeamSize;

        while (t < frames) {
            int steps = std::min(Q, frames - t);

            #pragma omp for
            for (size_t group = 0; group < n_groups; ++group) {
                auto subResult = beamSearch.run(emission, t, steps, rangeAdapter(*startHyp, group, n_groups));

                #pragma omp critical
                {
                    // need to save the individual beam's hyp_ because parent points into these arrays
                    for (int i = 1; i < steps; ++i)
                        result.hyp.emplace_back(std::move(subResult.hyp[i]));

                    // collect final hyps together for reduction
                    candidates.insert(candidates.end(),
                                      std::make_move_iterator(std::begin(subResult.hyp[steps])),
                                      std::make_move_iterator(std::end(subResult.hyp[steps])));

                    // get best score too
                    candidatesBestScore = std::max(candidatesBestScore, subResult.bestBeamScore);
                }
            }

            #pragma omp barrier
            #pragma omp single
            {
                std::vector<DecoderState> newHyp;
                beamSearchSelectBestCandidates(newHyp, candidates,
                                               candidatesBestScore - opt.beamThreshold, lm_, opt.beamSize);
                candidates.clear();
                result.hyp.emplace_back(std::move(newHyp));
                result.bestBeamScore = candidatesBestScore;
                candidatesBestScore = -INFINITY;
                startHyp = &result.hyp.back();
            }

            t += steps;
        }
    }

    return result;
}

//struct DiversityScoreAdjustment
//{
//    float extraNewTokenScore(int frame, int prevToken, int token)
//    {
//        return penalties[frame * N + token];
//    }
//    int N;
//    std::vector<float> penalties;
//};

//auto SimpleDecoder::diversity(const float *emissions,
//                              const int frames,
//                              const int nTokens) const
//    -> DecodeResult
//{
//    std::vector<std::vector<SimpleDecoderState>> hyp;
//    std::vector<SimpleDecoderState> best;
//    hyp.resize(1);

//    /* note: the lm reset itself with :start() */
//    hyp[0].emplace_back(
//                lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, sil_, -1);

//    int groups = 4;
//    DiversityScoreAdjustment adjuster{nTokens};
//    adjuster.penalties.resize(nTokens * frames, 0);

//    // essential to avoid reallocations
//    hyp.reserve((frames + 2) * groups);

//    for (int g = 0; g < groups; ++g) {
//        BeamSearch beamSearch{
//            .opt_ = opt_,
//            .transitions_ = transitions_,
//            .lexicon_ = lexicon_->getRoot(),
//            .lm_ = lm_,
//            .sil_ = sil_,
//            .unk_ = unk_,
//            .nTokens_ = nTokens,
//            .commands_ = commands_,
//        };
//        beamSearch.opt_.beamSize /= 4;

//        auto result = beamSearch.run(emissions, 0, frames, rangeAdapter(hyp[0]), adjuster);

//        // need to save the individual beam's hyp_ because parent points into these arrays
//        for (int i = 1; i < frames; ++i)
//            hyp.emplace_back(std::move(result.hyp[i]));

//        // save best beam
//        best.push_back(result.hyp[frames][0]);
//        std::cout << g << " " << toTokenString(&best.back()) << std::endl;

//        int i = frames - 1;
//        const SimpleDecoderState* it = &best.back();
//        while (it) {
//            int tok = it->lex->idx;
//            if (true || tok != sil_) {
//                for (int j = std::max(0, i - 2); j < frames && j <= i + 2; ++j)
//                    adjuster.penalties[j * nTokens + tok] -= 0.13;
//            }
//            it = it->parent;
//            i--;
//        }
//    }

//    std::vector<SimpleDecoderState> result;
//    beamSearchFinish(result, best, lm_, opt_);

//    return getHypothesis(&result[0], frames + 1);
//}
