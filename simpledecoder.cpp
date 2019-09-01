#include "criterion/CriterionUtils.h"

using namespace w2l;

#pragma pack(push, 1)
struct SimpleDecoderState {
  LMStatePtr lmState; // Language model state
  const FlatTrieNode* lex; // Trie node in the lexicon
  const SimpleDecoderState* parent; // Parent hypothesis
  /* tag represents bitwise:
   * int word : 23
   * bool prevBlank : 1
   * int token : 8
   */
  uint32_t tag;
  float score; // Score so far

  SimpleDecoderState(
      const LMStatePtr& lmState,
      const FlatTrieNode* lex,
      const SimpleDecoderState* parent,
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

  SimpleDecoderState()
      : lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
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
std::string toTokenString(const SimpleDecoderState *c)
{
    std::string curlex;
    for (auto it = c; it; it = it->parent) {
        if (it->getPrevBlank())
            curlex.insert(curlex.begin(), '_');
        else
            curlex.insert(curlex.begin(), token_lookup[it->lex->idx]);
    }
    return curlex;
};

static void beamSearchNewCandidate(
        std::vector<SimpleDecoderState> &candidates,
        float &bestScore,
        const float beamThreshold,
        const LMStatePtr& lmState,
        const FlatTrieNode* lex,
        const SimpleDecoderState* parent,
        const float score,
        const int token,
        const int word,
        const bool prevBlank = false)
{
    if (score < bestScore - beamThreshold)
        return;
    bestScore = std::max(bestScore, score);
    candidates.emplace_back(
            lmState, lex, parent, score, token, word, prevBlank);
}

// used for making an unordered_set of LexiconDecoderStates based on their LMStatePtr
struct StateHash {
    LMPtr lm_;
    size_t operator()(const SimpleDecoderState *v) const {
        return lm_->stateHash(v->lmState) ^ size_t(v->lex);
    }
};

struct StateEquality {
    LMPtr lm_;
    int operator()(const SimpleDecoderState *v1, const SimpleDecoderState *v2) const {
        return v1->lex == v2->lex && lm_->compareState(v1->lmState, v2->lmState) == 0;
    }
};

// Take at most beamSize items from candidates and fill nextHyp.
static void beamSearchSelectBestCandidates(
        std::vector<SimpleDecoderState>& nextHyp,
        std::vector<SimpleDecoderState>& candidates,
        const float scoreThreshold,
        const LMPtr &lm,
        const int beamSize)
{
    nextHyp.clear();
    nextHyp.reserve(std::min<size_t>(candidates.size(), beamSize));

    std::unordered_set<const SimpleDecoderState *, StateHash, StateEquality>
            seen(beamSize * 2, StateHash{lm}, StateEquality{lm});;

    std::make_heap(candidates.begin(), candidates.end());

    while (nextHyp.size() < beamSize && candidates.size() > 0) {
        auto& c = candidates[0];
        if (c.score < scoreThreshold) {
            break;
        }
        auto it = seen.find(&c);
        if (it == seen.end()) {
            nextHyp.emplace_back(std::move(c));
            seen.emplace(&nextHyp.back());
        }
        std::pop_heap(candidates.begin(), candidates.end());
        candidates.resize(candidates.size() - 1);
    }
}

// Wrap up by calling lmState->finish for all hyps
static void beamSearchFinish(
        std::vector<SimpleDecoderState> &hypOut,
        std::vector<SimpleDecoderState> &hypIn,
        const LMPtr &lm,
        const DecoderOptions &opt)
{
    float candidatesBestScore = -INFINITY;
    std::vector<SimpleDecoderState> candidates;
    candidates.reserve(hypIn.size());

    for (const SimpleDecoderState& prevHyp : hypIn) {
        const FlatTrieNode* prevLex = prevHyp.lex;
        const LMStatePtr& prevLmState = prevHyp.lmState;

        auto lmStateScorePair = lm->finish(prevLmState);
        beamSearchNewCandidate(
                    candidates,
                    candidatesBestScore,
                    opt.beamThreshold,
                    lmStateScorePair.first,
                    prevLex,
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

struct BeamSearch
{
    DecoderOptions opt_;
    const std::vector<float> &transitions_;
    const FlatTrieNode *lexicon_;
    LMPtr lm_;
    int sil_;
    int unk_;
    int nTokens_;
    const CommandModel &commands_;

    struct Result
    {
        std::vector<std::vector<SimpleDecoderState>> hyp;
        std::vector<SimpleDecoderState> ends;
        float bestBeamScore;
    };

    template <typename Range, typename Hooks = DefaultHooks>
    Result run(const float *emissions,
               const int startFrame,
               const int frames,
               Range initialHyp,
               Hooks &hooks = defaultBeamSearchHooks) const;
};

template <typename Range, typename Hooks = DefaultHooks>
auto BeamSearch::run(const float *emissions,
                     const int startFrame,
                     const int frames,
                     Range initialHyp,
                     Hooks &hooks) const
    -> BeamSearch::Result
{
    std::vector<std::vector<SimpleDecoderState>> hyp;
    hyp.resize(frames + 1, std::vector<SimpleDecoderState>());

    std::vector<SimpleDecoderState> ends;
    float endsBestScore = -INFINITY;

    std::vector<SimpleDecoderState> candidates;
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
            const auto &prevHyp = *hypIt;
            const FlatTrieNode* prevLex = prevHyp.lex;
            const LMStatePtr& prevLmState = prevHyp.lmState;
            const int prevIdx = prevHyp.getToken();
            bool hadIdenticalChild = false;

            if (!prevHyp.getPrevBlank()) {
            const float prevMaxScore =
                    prevLex == lexicon_ ? 0 : prevLex->maxScore; // just set root->maxScore to 0!
            /* (1) Try children */
            for (int iChild = 0 ; iChild < prevLex->nChildren; ++iChild) {
                auto lex = prevLex->child(iChild);
                int n = lex->idx;
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

                // We eat-up a new token
                if (lex->nChildren != 0) {
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_.beamThreshold,
                            prevLmState,
                            lex,
                            &prevHyp,
                            score + opt_.lmWeight * (lex->maxScore - prevMaxScore),
                            n,
                            -1
                            );
                }

                // If we got a true word
                for (int i = 0; i < lex->nLabel; i++) {
                    // could be a command word or a language word
                    int label = lex->label(i);
                    const bool wasCommand = label >= commands_.firstIdx;
                    if (wasCommand) {
                        auto &command = commands_.nodes.at(label);

                        const float optCommandScore = 1.5;
                        float commandLmScore = prevMaxScore; // not sure how we structure command-trie's maxScore yet
                        float commandScore = score + opt_.lmWeight * (commandLmScore - prevMaxScore) + optCommandScore;

                        assert(lex->nChildren == 0); // no further exploration on this lex
                        beamSearchNewCandidate(
                                candidates,
                                candidatesBestScore,
                                opt_.beamThreshold,
                                prevLmState, // TODO: Reset kenlm state!
                                lex,
                                &prevHyp,
                                commandScore,
                                n,
                                label
                                );
                    } else {
                        auto lmScoreReturn = lm_->score(prevLmState, label);
                        float lscore = score + opt_.lmWeight * (lmScoreReturn.second - prevMaxScore) + opt_.wordScore;

                        // Set up further language input
                        beamSearchNewCandidate(
                                candidates,
                                candidatesBestScore,
                                opt_.beamThreshold,
                                lmScoreReturn.first,
                                lexicon_,
                                &prevHyp,
                                lscore,
                                n,
                                label
                                );

                        // Allow command to follow
                        beamSearchNewCandidate(
                                candidates,
                                candidatesBestScore,
                                opt_.beamThreshold,
                                lmScoreReturn.first, // TODO: doesn't matter
                                commands_.tries->getRoot(),
                                &prevHyp,
                                lscore,
                                n,
                                label
                                );
                    }
                }

                // If we got an unknown word
                if (lex->nLabel == 0 && (opt_.unkScore > kNegativeInfinity)) {
                    auto lmScoreReturn = lm_->score(prevLmState, unk_);
                    beamSearchNewCandidate(
                            candidates,
                            candidatesBestScore,
                            opt_.beamThreshold,
                            lmScoreReturn.first,
                            lexicon_,
                            &prevHyp,
                            score + opt_.lmWeight * (lmScoreReturn.second - prevMaxScore) + opt_.unkScore,
                            n,
                            unk_
                            );
                }
            }
            }

            /* Try same lexicon node */
            if (!hadIdenticalChild) {
                int n = prevIdx;
                float score = prevHyp.score + emissions[frame * nTokens_ + n];
                if (frame > 0) {
                    score += transitions_[n * nTokens_ + prevIdx];
                }                
                bool blank = prevHyp.getPrevBlank() && n == sil_;
                if (n == sil_) {
                    float silEmission = emissions[frame * nTokens_ + n];
                    if (!blank && silEmission < maxEmissionScore) {
                        blank = true;
                    } else if (blank && silEmission >= maxEmissionScore) {
                        blank = false;
                    }

                    score += opt_.silWeight;
                }
                score += hooks.extraNewTokenScore(frame, prevIdx, n);

                beamSearchNewCandidate(
                        candidates,
                        candidatesBestScore,
                        opt_.beamThreshold,
                        prevLmState,
                        prevLex,
                        &prevHyp,
                        score,
                        n,
                        -1,
                        blank
                        );
            }
        }

        beamSearchSelectBestCandidates(hyp[t + 1], candidates, candidatesBestScore - opt_.beamThreshold, lm_, opt_.beamSize);
    }

    std::vector<SimpleDecoderState> filteredEnds;
    beamSearchSelectBestCandidates(filteredEnds, ends, endsBestScore - opt_.beamThreshold, lm_, opt_.beamSize);

    return Result{std::move(hyp), std::move(filteredEnds), candidatesBestScore};
}

struct SimpleDecoder
{
    DecoderOptions opt_;
    FlatTriePtr lexicon_;
    LMPtr lm_;
    int sil_;
    int unk_;
    std::vector<float> transitions_;
    const CommandModel commands_;

    DecodeResult normal(const float *emissions,
                        const int frames,
                        const int nTokens,
                        const FlatTrieNode *startTrie = nullptr) const;

    DecodeResult groupThreading(const float *emissions,
                                const int frames,
                                const int nTokens) const;

    DecodeResult diversity(const float *emissions,
                           const int frames,
                           const int nTokens) const;
};

auto SimpleDecoder::normal(const float *emissions,
                           const int frames,
                           const int nTokens,
                           const FlatTrieNode *startTrie) const
    -> DecodeResult
{
    std::vector<std::vector<SimpleDecoderState>> hyp;
    hyp.resize(1);

    /* note: the lm reset itself with :start() */
    if (!startTrie)
        startTrie = lexicon_->getRoot();
    hyp[0].emplace_back(
                lm_->start(0), startTrie, nullptr, 0.0, sil_, -1);

    BeamSearch beamSearch{
        .opt_ = opt_,
        .transitions_ = transitions_,
        .lexicon_ = lexicon_->getRoot(),
        .lm_ = lm_,
        .sil_ = sil_,
        .unk_ = unk_,
        .nTokens_ = nTokens,
        .commands_ = commands_,
    };

    auto beams = beamSearch.run(emissions, 0, frames, rangeAdapter(hyp[0]));

    std::vector<SimpleDecoderState> result;
    beamSearchFinish(result, beams.hyp.back(), lm_, opt_);

    return getHypothesis(&result[0], beams.hyp.size());
}

auto SimpleDecoder::groupThreading(const float *emissions,
                                   const int frames,
                                   const int nTokens) const
    -> DecodeResult
{
    std::vector<std::vector<SimpleDecoderState>> hyp;
    hyp.resize(1);

    /* note: the lm reset itself with :start() */
    hyp[0].emplace_back(
                lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, sil_, -1);

    // Run Q-steps of beamseach on subgroups of hyp
    int Q = 5;
    int n_groups = 4;

    // essential to avoid reallocations
    hyp.reserve((frames + 2) * n_groups);

    std::vector<SimpleDecoderState> candidates;
    float candidatesBestScore = -INFINITY;

    std::vector<SimpleDecoderState> *startHyp = &hyp.back();

    #pragma omp parallel num_threads(4)
    {
        int t = 0;

        BeamSearch beamSearch{
            .opt_ = opt_,
            .transitions_ = transitions_,
            .lexicon_ = lexicon_->getRoot(),
            .lm_ = lm_,
            .sil_ = sil_,
            .unk_ = unk_,
            .nTokens_ = nTokens,
            .commands_ = commands_,
        };
        beamSearch.opt_.beamSize /= 4;

        while (t < frames) {
            int steps = std::min(Q, frames - t);

            #pragma omp for
            for (size_t group = 0; group < n_groups; ++group) {
                auto result = beamSearch.run(emissions, t, steps, rangeAdapter(*startHyp, group, n_groups));

                #pragma omp critical
                {
                    // need to save the individual beam's hyp_ because parent points into these arrays
                    for (int i = 1; i < steps; ++i)
                        hyp.emplace_back(std::move(result.hyp[i]));

                    // collect final hyps together for reduction
                    candidates.insert(candidates.end(),
                                      std::make_move_iterator(std::begin(result.hyp[steps])),
                                      std::make_move_iterator(std::end(result.hyp[steps])));

                    // get best score too
                    candidatesBestScore = std::max(candidatesBestScore, result.bestBeamScore);
                }
            }

            #pragma omp barrier
            #pragma omp single
            {
                std::vector<SimpleDecoderState> newHyp;
                beamSearchSelectBestCandidates(newHyp, candidates,
                                               candidatesBestScore - opt_.beamThreshold, lm_, opt_.beamSize);
                candidates.clear();
                candidatesBestScore = -INFINITY;
                hyp.emplace_back(std::move(newHyp));
                startHyp = &hyp.back();
            }

            t += steps;
        }
    }

    std::vector<SimpleDecoderState> result;
    beamSearchFinish(result, hyp.back(), lm_, opt_);

    return getHypothesis(&result[0], frames + 1);
}

struct DiversityScoreAdjustment
{
    float extraNewTokenScore(int frame, int prevToken, int token)
    {
        return penalties[frame * N + token];
    }
    int N;
    std::vector<float> penalties;
};

auto SimpleDecoder::diversity(const float *emissions,
                              const int frames,
                              const int nTokens) const
    -> DecodeResult
{
    std::vector<std::vector<SimpleDecoderState>> hyp;
    std::vector<SimpleDecoderState> best;
    hyp.resize(1);

    /* note: the lm reset itself with :start() */
    hyp[0].emplace_back(
                lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, sil_, -1);

    int groups = 4;
    DiversityScoreAdjustment adjuster{nTokens};
    adjuster.penalties.resize(nTokens * frames, 0);

    // essential to avoid reallocations
    hyp.reserve((frames + 2) * groups);

    for (int g = 0; g < groups; ++g) {
        BeamSearch beamSearch{
            .opt_ = opt_,
            .transitions_ = transitions_,
            .lexicon_ = lexicon_->getRoot(),
            .lm_ = lm_,
            .sil_ = sil_,
            .unk_ = unk_,
            .nTokens_ = nTokens,
            .commands_ = commands_,
        };
        beamSearch.opt_.beamSize /= 4;

        auto result = beamSearch.run(emissions, 0, frames, rangeAdapter(hyp[0]), adjuster);

        // need to save the individual beam's hyp_ because parent points into these arrays
        for (int i = 1; i < frames; ++i)
            hyp.emplace_back(std::move(result.hyp[i]));

        // save best beam
        best.push_back(result.hyp[frames][0]);
        std::cout << g << " " << toTokenString(&best.back()) << std::endl;

        int i = frames - 1;
        const SimpleDecoderState* it = &best.back();
        while (it) {
            int tok = it->lex->idx;
            if (true || tok != sil_) {
                for (int j = std::max(0, i - 2); j < frames && j <= i + 2; ++j)
                    adjuster.penalties[j * nTokens + tok] -= 0.13;
            }
            it = it->parent;
            i--;
        }
    }

    std::vector<SimpleDecoderState> result;
    beamSearchFinish(result, best, lm_, opt_);

    return getHypothesis(&result[0], frames + 1);
}
