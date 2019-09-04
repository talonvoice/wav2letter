#include "criterion/CriterionUtils.h"

using namespace w2l;

namespace KenFlatTrieLM {
    // A single LM reference gets passed around. It's for context data.
    struct LM {
        LMPtr ken;
        FlatTriePtr trie;
        int firstCommandLabel;
    };

    // Every DecoderState will store a State.
    struct State {
        LMStatePtr kenState;
        const FlatTrieNode *lex = nullptr;

        // used for making an unordered_set of const State*
        struct Hash {
            const LM &lm_;
            size_t operator()(const State *v) const {
                return lm_.ken->stateHash(v->kenState) ^ size_t(v->lex);
            }
        };

        struct Equality {
            const LM &lm_;
            int operator()(const State *v1, const State *v2) const {
                return v1->lex == v2->lex && lm_.ken->compareState(v1->kenState, v2->kenState) == 0;
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
                    if (label < lm.firstCommandLabel) {
                        auto kenAndScore = lm.ken->score(base.kenState, label);
                        State it;
                        it.kenState = std::move(kenAndScore.first);
                        it.lex = lm.trie->getRoot();
                        fn(std::move(it), label, kenAndScore.second);
                    } else {
                        fn(actualize(), label, 1.5);
                    }
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
        void forChildren(Fn&& fn) const {
            const auto n = lex->nChildren;
            for (int i = 0; i < n; ++i) {
                auto nlex = lex->child(i);
                fn(Proxy{*this, nlex}, nlex->idx, nlex->nChildren > 0);
            }
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
    int16_t token = tag & 0xFF;
    return token == 0xFF ? -1 : token;
  }
  void setToken(int token) {
    tag = (tag & ~0xFF) | (token & 0xFF);
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
};
#pragma pack(pop)

static std::string token_lookup("|'abcdefghijklmnopqrstuvwxyz");
std::string toTokenString(const SimpleDecoderState<KenFlatTrieLM::State> *c)
{
    std::string curlex;
    for (auto it = c; it; it = it->parent) {
        if (it->getPrevBlank())
            curlex.insert(curlex.begin(), '_');
        else
            curlex.insert(curlex.begin(), token_lookup[it->getToken()]);
    }
    return curlex;
};

template <typename LMStateType1, typename LMStateType2>
static void beamSearchNewCandidate(
        std::vector<SimpleDecoderState<LMStateType2>> &candidates,
        float &bestScore,
        const float beamThreshold,
        LMStateType1 lmState,
        const SimpleDecoderState<LMStateType2>* parent,
        const float score,
        const int token,
        const int word,
        const bool prevBlank = false)
{
    if (score < bestScore - beamThreshold)
        return;
    bestScore = std::max(bestScore, score);
    candidates.emplace_back(
            std::move(lmState.actualize()), parent, score, token, word, prevBlank);
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
                    opt.beamThreshold,
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
struct DefaultHooks
{
    float extraNewTokenScore(int frame, int prevToken, int token)
    {
        return 0;
    }
};
DefaultHooks defaultBeamSearchHooks;

struct CommandModel
{
    struct Node
    {
        bool allowLanguage;
        const FlatTrieNode *next;
    };

    int firstIdx;
    FlatTriePtr tries; // should have multiple roots
    std::unordered_map<size_t, Node> nodes; // map word idx to what comes after
};

template <typename LM, typename LMStateType>
struct BeamSearch
{
    using DecoderState = SimpleDecoderState<LMStateType>;

    DecoderOptions opt_;
    const std::vector<float> &transitions_;
    const LM &lm_;
    int sil_;
    int unk_;
    int nTokens_;

    struct Result
    {
        std::vector<std::vector<DecoderState>> hyp;
        std::vector<DecoderState> ends;
        float bestBeamScore;
    };

    template <typename Range, typename Hooks = DefaultHooks>
    Result run(const float *emissions,
               const int startFrame,
               const int frames,
               Range initialHyp,
               Hooks &hooks = defaultBeamSearchHooks) const;
};

template <typename LM, typename LMStateType>
template <typename Range, typename Hooks>
auto BeamSearch<LM, LMStateType>::run(
        const float *emissions,
        const int startFrame,
        const int frames,
        Range initialHyp,
        Hooks &hooks) const
    -> Result
{
    std::vector<std::vector<DecoderState>> hyp;
    hyp.resize(frames + 1, std::vector<DecoderState>());

    std::vector<DecoderState> ends;
    float endsBestScore = -INFINITY;

    std::vector<DecoderState> candidates;
    candidates.reserve(opt_.beamSize);
    float candidatesBestScore = kNegativeInfinity;

    for (int t = 0; t < frames; t++) {
        int frame = startFrame + t;
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

        auto range = t == 0 ? initialHyp : rangeAdapter(hyp[t]);
        while (auto hypIt = range()) {
            const auto& prevHyp = *hypIt;
            const auto& prevLmState = prevHyp.lmState;
            const int prevIdx = prevHyp.getToken();
            bool hadIdenticalChild = false;

            const float prevMaxScore = prevLmState.maxWordScore();
            /* (1) Try children */
            prevLmState.forChildren([&, prevIdx, prevMaxScore](typename LMStateType::Proxy lmState, int n, bool hasChildren) {
                if (n == prevIdx)
                    hadIdenticalChild = true;
                float score = prevHyp.score + emissions[frame * nTokens_ + n];
                if (frame > 0) {
                    score += transitions_[n * nTokens_ + prevIdx];
                }
                if (n == sil_) {
                    score += opt_.silWeight;
                }
                score += hooks.extraNewTokenScore(frame, prevIdx, n);

                // If we got a true word
                bool hadLabel = false;
                lmState.forLabels(lm_, [&, prevMaxScore, score](LMStateType labelLmState, int label, float lmScore) {
                    hadLabel = true;
                    float lScore = score + opt_.lmWeight * (lmScore - prevMaxScore) + opt_.wordScore;
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_.beamThreshold,
                            std::move(labelLmState),
                            &prevHyp,
                            lScore,
                            n,
                            label
                            );
                });

                // We eat-up a new token
                if (hasChildren) {
                    float lScore = score + opt_.lmWeight * (lmState.maxWordScore() - prevMaxScore);
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_.beamThreshold,
                            std::move(lmState),
                            &prevHyp,
                            lScore,
                            n,
                            -1
                            );
                }

                // If we got an unknown word
                if (!hadLabel && (opt_.unkScore > kNegativeInfinity)) {
//                    auto lmScoreReturn = lm_->score(prevLmState, unk_);
//                    beamSearchNewCandidate(
//                            candidates,
//                            candidatesBestScore,
//                            opt_.beamThreshold,
//                            lmScoreReturn.first,
//                            lexicon_,
//                            &prevHyp,
//                            score + opt_.lmWeight * (lmScoreReturn.second - prevMaxScore) + opt_.unkScore,
//                            n,
//                            unk_
//                            );
                }
            });

            /* Try same lexicon node */
            if (!hadIdenticalChild) {
                int n = prevIdx;
                float score = prevHyp.score + emissions[frame * nTokens_ + n];
                if (frame > 0) {
                    score += transitions_[n * nTokens_ + prevIdx];
                }                
                if (n == sil_) {
                    score += opt_.silWeight;
                }
                score += hooks.extraNewTokenScore(frame, prevIdx, n);

                beamSearchNewCandidate(
                        candidates,
                        candidatesBestScore,
                        opt_.beamThreshold,
                        prevLmState,
                        &prevHyp,
                        score,
                        n,
                        -1
                        );
            }
        }

        beamSearchSelectBestCandidates(hyp[t + 1], candidates, candidatesBestScore - opt_.beamThreshold, lm_, opt_.beamSize);
    }

    std::vector<DecoderState> filteredEnds;
    beamSearchSelectBestCandidates(filteredEnds, ends, endsBestScore - opt_.beamThreshold, lm_, opt_.beamSize);

    return Result{std::move(hyp), std::move(filteredEnds), candidatesBestScore};
}

template <typename LM, typename LMStateType>
struct SimpleDecoder
{
    using DecoderState = SimpleDecoderState<LMStateType>;

    DecoderOptions opt_;
    LM lm_;
    int sil_;
    int unk_;
    std::vector<float> transitions_;

    DecodeResult normal(const float *emissions,
                        const int frames,
                        const int nTokens,
                        const LMStateType startState) const;

//    DecodeResult groupThreading(const float *emissions,
//                                const int frames,
//                                const int nTokens) const;

//    DecodeResult diversity(const float *emissions,
//                           const int frames,
//                           const int nTokens) const;
};

template <typename LM, typename LMStateType>
auto SimpleDecoder<LM, LMStateType>::normal(
        const float *emissions,
        const int frames,
        const int nTokens,
        const LMStateType startState) const
    -> DecodeResult
{
    std::vector<std::vector<DecoderState>> hyp;
    hyp.resize(1);

    /* note: the lm reset itself with :start() */
    hyp[0].emplace_back(
                startState, nullptr, 0.0, sil_, -1);

    BeamSearch<LM, LMStateType> beamSearch{
        .opt_ = opt_,
        .transitions_ = transitions_,
        .lm_ = lm_,
        .sil_ = sil_,
        .unk_ = unk_,
        .nTokens_ = nTokens,
    };

    auto beams = beamSearch.run(emissions, 0, frames, rangeAdapter(hyp[0]));

    std::vector<DecoderState> result;
    beamSearchFinish(result, beams.hyp.back(), lm_, opt_);

    return getHypothesis(&result[0], beams.hyp.size());
}

//auto SimpleDecoder::groupThreading(const float *emissions,
//                                   const int frames,
//                                   const int nTokens) const
//    -> DecodeResult
//{
//    std::vector<std::vector<SimpleDecoderState>> hyp;
//    hyp.resize(1);

//    /* note: the lm reset itself with :start() */
//    hyp[0].emplace_back(
//                lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, sil_, -1);

//    // Run Q-steps of beamseach on subgroups of hyp
//    int Q = 5;is
//    int n_groups = 4;

//    // essential to avoid reallocations
//    hyp.reserve((frames + 2) * n_groups);

//    std::vector<SimpleDecoderState> candidates;
//    float candidatesBestScore = -INFINITY;

//    std::vector<SimpleDecoderState> *startHyp = &hyp.back();

//    #pragma omp parallel num_threads(4)
//    {
//        int t = 0;

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

//        while (t < frames) {
//            int steps = std::min(Q, frames - t);

//            #pragma omp for
//            for (size_t group = 0; group < n_groups; ++group) {
//                auto result = beamSearch.run(emissions, t, steps, rangeAdapter(*startHyp, group, n_groups));

//                #pragma omp critical
//                {
//                    // need to save the individual beam's hyp_ because parent points into these arrays
//                    for (int i = 1; i < steps; ++i)
//                        hyp.emplace_back(std::move(result.hyp[i]));

//                    // collect final hyps together for reduction
//                    candidates.insert(candidates.end(),
//                                      std::make_move_iterator(std::begin(result.hyp[steps])),
//                                      std::make_move_iterator(std::end(result.hyp[steps])));

//                    // get best score too
//                    candidatesBestScore = std::max(candidatesBestScore, result.bestBeamScore);
//                }
//            }

//            #pragma omp barrier
//            #pragma omp single
//            {
//                std::vector<SimpleDecoderState> newHyp;
//                beamSearchSelectBestCandidates(newHyp, candidates,
//                                               candidatesBestScore - opt_.beamThreshold, lm_, opt_.beamSize);
//                candidates.clear();
//                candidatesBestScore = -INFINITY;
//                hyp.emplace_back(std::move(newHyp));
//                startHyp = &hyp.back();
//            }

//            t += steps;
//        }
//    }

//    std::vector<SimpleDecoderState> result;
//    beamSearchFinish(result, hyp.back(), lm_, opt_);

//    return getHypothesis(&result[0], frames + 1);
//}

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
