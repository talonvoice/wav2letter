#include <iostream>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <cassert>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils-base.h"
#include "decoder/Utils.h"
#include "decoder/Trie.h"
#include "decoder/WordLMDecoder.h"
#include "decoder/TokenLMDecoder.h"
#include "lm/KenLM.h"

#include "w2l.h"
#include "w2l_p.h"

#include "simpledecoder.cpp"

namespace w2l {
// from common/Utils.h (which includes flashlight, so we don't include it)
std::string join(const std::string& delim, const std::vector<std::string>& vec);
}

template <typename T>
std::vector<T> afToVector(const af::array& arr) {
  std::vector<T> vec(arr.elements());
  arr.host(vec.data());
  return vec;
}

w2l_decode_options w2l_decode_defaults {
    2500,
    25,
    1.0,
    1.0,
    -INFINITY,
    false,
    0.0,
};

using namespace w2l;

DecoderOptions toW2lDecoderOptions(const w2l_decode_options &opts) {
    return DecoderOptions(
                opts.beamsize,
                opts.beamthresh,
                opts.lmweight,
                opts.wordscore,
                opts.unkweight,
                opts.logadd,
                opts.silweight,
                CriterionType::ASG);
}

class WrapDecoder {
public:
    WrapDecoder(Engine *engine, const char *languageModelPath, const char *lexiconPath, const char *flattriePath, const w2l_decode_options *opts) {
        tokenDict = engine->tokenDict;
        silIdx = tokenDict.getIndex(kSilToken);

        auto lexicon = loadWords(lexiconPath, -1);

        // Adjust the lexicon words to always end in silence
        for (auto &entry : lexicon) {
            for (auto &spelling : entry.second) {
                if (spelling.empty() || spelling.back() != "|")
                    spelling.push_back("|");
            }
        }

        wordDict = createWordDict(lexicon);
        lm = std::make_shared<KenLM>(languageModelPath, wordDict);

        // Load the trie
        std::ifstream flatTrieIn(flattriePath);
        flatTrie = std::make_shared<FlatTrie>();
        size_t byteSize;
        flatTrieIn >> byteSize;
        flatTrie->storage.resize(byteSize / 4);
        flatTrieIn.read(reinterpret_cast<char *>(flatTrie->storage.data()), byteSize);

        // the root maxScore should be 0 during search and it's more convenient to set here
        const_cast<FlatTrieNode *>(flatTrie->getRoot())->maxScore = 0;

        CriterionType criterionType = CriterionType::ASG;
        if (engine->criterionType == kCtcCriterion) {
            criterionType = CriterionType::CTC;
        } else if (engine->criterionType != kAsgCriterion) {
            // FIXME:
            LOG(FATAL) << "[Decoder] Invalid model type: " << engine->criterionType;
        }
        decoderOpt = toW2lDecoderOptions(*opts);

        KenFlatTrieLM::LM lmWrap;
        lmWrap.ken = lm;
        lmWrap.trie = flatTrie;

        auto transition = engine->transitions();
        decoder.reset(new SimpleDecoder<KenFlatTrieLM::LM, KenFlatTrieLM::State>{
            decoderOpt,
            lmWrap,
            silIdx,
            wordDict.getIndex(kUnkToken),
            transition});
    }
    ~WrapDecoder() {}

    DecodeResult decode(Emission *emission) {
        auto rawEmission = emission->emission;
        auto emissionVec = afToVector<float>(rawEmission);
        int N = rawEmission.dims(0);
        int T = rawEmission.dims(1);

        std::vector<float> score;
        std::vector<std::vector<int>> wordPredictions;
        std::vector<std::vector<int>> letterPredictions;
        KenFlatTrieLM::State startState;
        startState.lex = flatTrie->getRoot();
        startState.kenState = lm->start(0);
        return decoder->normal(emissionVec.data(), T, N, startState);
        //return decoder->groupThreading(emissionVec.data(), T, N);
    }

    char *resultWords(const DecodeResult &result) {
        auto rawWordPrediction = validateIdx(result.words, wordDict.getIndex(kUnkToken));
        auto wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict);
        auto words = join(" ", wordPrediction);
        return strdup(words.c_str());
    }

    char *resultTokens(const DecodeResult &result) {
        auto tknIdx = result.tokens;

        // ends with a -1 token, make into silence instead
        // tknIdx2Ltr will filter out the first and last if they are silences
        if (tknIdx.size() > 0 && tknIdx.back() == -1)
           tknIdx.back() = silIdx;

        auto tknLtrs = tknIdx2Ltr(tknIdx, tokenDict);
        std::string out;
        for (const auto &ltr : tknLtrs)
            out.append(ltr);
        return strdup(out.c_str());
    }

    std::shared_ptr<KenLM> lm;
    FlatTriePtr flatTrie;
    std::unique_ptr<SimpleDecoder<KenFlatTrieLM::LM, KenFlatTrieLM::State>> decoder;
    Dictionary wordDict;
    Dictionary tokenDict;
    DecoderOptions decoderOpt;
    int silIdx;
};

namespace DFALM {

// flags on dfa nodes
enum {
    FLAG_NONE    = 0,
    FLAG_TERM    = 1,
};

// special token values on dfa edges
enum {
    TOKEN_LMWORD     = 255,
    TOKEN_LMWORD_CTX = 254,
};

struct LM {
    LMPtr ken;
    FlatTriePtr trie;
    const w2l_dfa_node *dfa;
    int firstCommandLabel = 0;

    const w2l_dfa_node *get(const w2l_dfa_node *base, const int32_t idx) const {
        return reinterpret_cast<const w2l_dfa_node *>(reinterpret_cast<const uint8_t *>(base) + idx);
    }

    float commandScore = 1.5;
};

struct State {
    // pos in the grammar, never null
    const w2l_dfa_node *grammarLex = nullptr;
    // pos in trie, only set while decoding a lexicon word
    const FlatTrieNode *dictLex = nullptr;
    // ken state, preserved even when dictLex goes nullptr
    LMStatePtr kenState = nullptr;
    // whether the last edge in the grammar was a silToken.
    // could be optimized away
    bool wordEnd = false;

    // used for making an unordered_set of const State*
    struct Hash {
        const LM &unused;
        size_t operator()(const State *v) const {
            return std::hash<const void*>()(v->grammarLex) ^ std::hash<const void*>()(v->dictLex);
        }
    };

    struct Equality {
        const LM &lm_;
        int operator()(const State *v1, const State *v2) const {
            return v1->grammarLex == v2->grammarLex && v1->dictLex == v2->dictLex && lm_.ken->compareState(v1->kenState, v2->kenState) == 0;
        }
    };

    // Iterate over labels, calling fn with: the new State, the label index and the lm score
    template <typename Fn>
    void forLabels(const LM &lm, Fn&& fn) const {
        // in dictionary mode we may return positive labels for dictionary words
        if (dictLex) {
            const auto n = dictLex->nLabel;
            for (int i = 0; i < n; ++i) {
                int label = dictLex->label(i);
                auto kenAndScore = lm.ken->score(kenState, label);
                State it;
                it.grammarLex = grammarLex;
                it.dictLex = nullptr;
                it.kenState = std::move(kenAndScore.first);
                fn(std::move(it), label, kenAndScore.second);
            }
            return;
        }

        // command labels are offsets from lm.dfa, plus the firstCommandLabel value
        if (wordEnd) {
            fn(*this, lm.firstCommandLabel + (reinterpret_cast<const uint8_t*>(grammarLex) - reinterpret_cast<const uint8_t*>(lm.dfa)), lm.commandScore);
        }
    }

    // Call finish() on the lm, like for end-of-sentence scoring
    std::pair<State, float> finish(const LM &lm) const {
        bool bad = dictLex || !(grammarLex->flags & FLAG_TERM);
        return {*this, bad ? -1000000 : 0};
    }

    float maxWordScore() const {
        return 0; // could control whether the beam search gets scores before finishing commands
    }

    // Iterate over children of the state, calling fn with:
    // new State, new token index and whether the new state has children
    template <typename Fn>
    bool forChildren(int frame, const LM &lm, Fn&& fn) const {
        // If a dictionary word was started only consider its trie children.
        if (dictLex) {
            const auto n = dictLex->nChildren;
            for (int i = 0; i < n; ++i) {
                auto nlex = dictLex->child(i);
                fn(State{grammarLex, nlex, kenState}, nlex->idx, nlex->nChildren > 0);
            }
            return true;
        }

        // Otherwise look at the grammar dfa
        for (int i = 0; i < grammarLex->nEdges; ++i) {
            const auto &edge = grammarLex->edges[i];
            auto nlex = lm.get(grammarLex, edge.offset);

            // For dictionary edges start exploring the trie
            if (edge.token == TOKEN_LMWORD || edge.token == TOKEN_LMWORD_CTX) {
                auto nextKenState = edge.token == TOKEN_LMWORD_CTX ? kenState : nullptr;
                if (!nextKenState)
                    nextKenState = lm.ken->start(0);
                auto dictRoot = lm.trie->getRoot();
                const auto n = dictRoot->nChildren;
                for (int i = 0; i < n; ++i) {
                    auto nDictLex = dictRoot->child(i);
                    fn(State{nlex, nDictLex, nextKenState}, nDictLex->idx, nDictLex->nChildren > 0);
                }
            } else {
                fn(State{nlex, nullptr, nullptr, edge.token == 0}, edge.token, edge.token != 0);
            }
        }
        return true;
    }

    State &actualize() {
        return *this;
    }
};

} // namespace DFALM

using CombinedDecoder = SimpleDecoder<DFALM::LM, DFALM::State>;

// Score adjustment during beam search to reject beams early
// that diverge too much from the best emission-transmission score.
struct ViterbiDifferenceRejecter {
    std::vector<int> viterbiToks;

    // index i contains the emission-transmission score of up to windowMaxSize
    // previous frames of the viterbiTokens, see precomputeViterbiWindowScores.
    std::vector<float> viterbiWindowScores;

    int windowMaxSize;
    float threshold;
    float *emissions;
    float *transitions;
    int T;

    float extraNewTokenScore(int frame, const CombinedDecoder::DecoderState &prevState, int token) const {
        auto refScore = viterbiWindowScores[frame];

        bool allSilence = token == 0;
        int prevToken = token;
        auto thisState = &prevState;
        float thisScore = emissions[frame * T + token];
        int thisWindow = 1;
        while (thisWindow < windowMaxSize && thisState && frame - thisWindow >= 0) {
            token = thisState->getToken();
            if (token != 0)
                allSilence = false;
            thisScore += emissions[(frame - thisWindow) * T + token] + transitions[prevToken * T + token];
            ++thisWindow;
            prevToken = token;
            thisState = thisState->parent;
        }

        // rejecting based on non-full windows is too unstable, wait for full window
        if (thisWindow < windowMaxSize)
            return 0;

        if (thisScore / refScore < threshold) {
            return -100000;
        }

        // Only allow a full silence window if the viterbi tok in the middle is also silence
        if (allSilence && viterbiToks[frame - windowMaxSize/2] != 0) {
            return -100000;
        }

        return 0;
    }

    void precomputeViterbiWindowScores(int segStart, const std::vector<int> &viterbiToks) {
        float score = 0;
        const int N = viterbiToks.size();
        for (int j = segStart; j < N; ++j) {
            score += emissions[(j - segStart) * T + viterbiToks[j]];
            if (j != segStart)
                score += transitions[viterbiToks[j] * T + viterbiToks[j - 1]];
            viterbiWindowScores.push_back(score);
            if (j - segStart < windowMaxSize - 1)
                continue;
            auto r = j - (windowMaxSize - 1);
            score -= emissions[(r - segStart) * T + viterbiToks[r]];
            if (r != segStart)
                score -= transitions[viterbiToks[r] * T + viterbiToks[r - 1]];
        }
    }
};

extern "C" {

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_emission w2l_emission;
typedef struct w2l_decoderesult w2l_decoderesult;

w2l_engine *w2l_engine_new(const char *acoustic_model_path, const char *tokens_path) {
    // TODO: what other engine config do I need?
    auto engine = new Engine(acoustic_model_path, tokens_path);
    return reinterpret_cast<w2l_engine *>(engine);
}

w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count) {
    auto emission = reinterpret_cast<Engine *>(engine)->process(samples, sample_count);
    return reinterpret_cast<w2l_emission *>(emission);
}

bool w2l_engine_export(w2l_engine *engine, const char *path) {
    return reinterpret_cast<Engine *>(engine)->exportModel(path);
}

void w2l_engine_free(w2l_engine *engine) {
    if (engine)
        delete reinterpret_cast<Engine *>(engine);
}

char *w2l_emission_text(w2l_emission *emission) {
    return reinterpret_cast<Emission *>(emission)->text();
}

float *w2l_emission_values(w2l_emission *emission, int *frames, int *tokens) {
    auto em = reinterpret_cast<Emission *>(emission);
    auto data = afToVector<float>(em->emission);
    *frames = em->emission.dims(1);
    *tokens = em->emission.dims(0);
    int datasize = sizeof(float) * *frames * *tokens;
    float *out = static_cast<float *>(malloc(datasize));
    memcpy(out, data.data(), datasize);
    return out;
}

void w2l_emission_free(w2l_emission *emission) {
    if (emission)
        delete reinterpret_cast<Emission *>(emission);
}

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path, const char *flattrie_path, const w2l_decode_options *opts) {
    // TODO: what other config? beam size? smearing? lm weight?
    auto decoder = new WrapDecoder(reinterpret_cast<Engine *>(engine), kenlm_model_path, lexicon_path, flattrie_path, opts);
    return reinterpret_cast<w2l_decoder *>(decoder);
}

w2l_decoderesult *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission) {
    auto result = new DecodeResult(reinterpret_cast<WrapDecoder *>(decoder)->decode(reinterpret_cast<Emission *>(emission)));
    return reinterpret_cast<w2l_decoderesult *>(result);
}

char *w2l_decoder_result_words(w2l_decoder *decoder, w2l_decoderesult *decoderesult) {
    auto decoderObj = reinterpret_cast<WrapDecoder *>(decoder);
    auto result = reinterpret_cast<DecodeResult *>(decoderesult);
    return decoderObj->resultWords(*result);
}

char *w2l_decoder_result_tokens(w2l_decoder *decoder, w2l_decoderesult *decoderesult) {
    auto decoderObj = reinterpret_cast<WrapDecoder *>(decoder);
    auto result = reinterpret_cast<DecodeResult *>(decoderesult);
    return decoderObj->resultTokens(*result);
}

void w2l_decoderesult_free(w2l_decoderesult *decoderesult) {
    if (decoderesult)
        delete reinterpret_cast<DecodeResult *>(decoderesult);
}

void w2l_decoder_free(w2l_decoder *decoder) {
    if (decoder)
        delete reinterpret_cast<WrapDecoder *>(decoder);
}

char *w2l_decoder_dfa(w2l_engine *engine, w2l_decoder *decoder, w2l_emission *emission, w2l_dfa_node *dfa, w2l_dfa_decode_options *opts) {
    auto engineObj = reinterpret_cast<Engine *>(engine);
    auto decoderObj = reinterpret_cast<WrapDecoder *>(decoder);
    auto emissionObj = reinterpret_cast<Emission *>(emission);
    auto rawEmission = emissionObj->emission;

    auto emissionVec = afToVector<float>(rawEmission);
    int T = rawEmission.dims(0);
    int N = rawEmission.dims(1);
    auto &transitions = decoderObj->decoder->transitions_;

    auto tokensToString = [engineObj](const std::vector<int> &tokens, int from, int to) {
        std::string out;
        for (int i = from; i < to; ++i)
            out.append(engineObj->tokenDict.getEntry(tokens[i]));
        return out;
    };
    auto tokensToStringDedup = [engineObj](const std::vector<int> &tokens, int from, int to) {
        std::string out;
        int tok = -1;
        for (int i = from; i < to; ++i) {
            if (tok == tokens[i])
                continue;
            tok = tokens[i];
            out.append(engineObj->tokenDict.getEntry(tok));
        }
        return out;
    };

    auto viterbiToks =
        afToVector<int>(engineObj->viterbiPath(rawEmission));
    assert(N == viterbiToks.size());

    // Lets skip decoding if viterbi thinks it's all silence
    bool allSilence = true;
    for (auto t : viterbiToks) {
        if (t != 0)
            allSilence = false;
    }
    if (allSilence)
        return nullptr;

    auto dfalm = DFALM::LM{decoderObj->lm, decoderObj->flatTrie, dfa};
    dfalm.commandScore = opts->command_score;
    dfalm.firstCommandLabel = decoderObj->wordDict.indexSize();

    auto commandDecoder = CombinedDecoder{
                decoderObj->decoderOpt,
                dfalm,
                decoderObj->silIdx,
                decoderObj->wordDict.getIndex(kUnkToken),
                transitions};

    if (opts->debug) {
        std::cout << "detecting in viterbi toks: " << tokensToString(viterbiToks, 0, viterbiToks.size()) << std::endl;
    }

    auto appendSpaced = [&](const std::string &base, const std::string &str, bool command = false) {
        std::string out = base;
        if (!out.empty())
            out += " ";
        if (command)
            out += "@";
        out += str;
        return out;
    };

    ViterbiDifferenceRejecter rejecter;
    rejecter.windowMaxSize = opts->rejection_window_frames;
    rejecter.threshold = opts->rejection_threshold;
    rejecter.emissions = emissionVec.data();
    rejecter.transitions = transitions.data();
    rejecter.T = T;
    rejecter.precomputeViterbiWindowScores(0, viterbiToks);
    rejecter.viterbiToks = viterbiToks;

    DFALM::State commandState;
    commandState.grammarLex = dfalm.dfa;

    // in the future we could stop the decode after one word instead of
    // decoding everything
    int decodeLen = N;
    std::vector<CombinedDecoder::DecoderState> startStates;
    startStates.emplace_back(commandState, nullptr, 0.0, 0, -1);
    auto unfinishedBeams = commandDecoder.normalAll(emissionVec.data(), decodeLen, T, startStates, rejecter);

    // Finishing kills beams that end in the middle of a word, or
    // in a grammar state that isn't TERM
    std::vector<CombinedDecoder::DecoderState> beamEnds;
    beamSearchFinish(beamEnds, unfinishedBeams.hyp.back(), dfalm, commandDecoder.opt_);

    if (beamEnds.empty())
        return nullptr;

    if (opts->debug) {
        for (const auto &beamEnd : beamEnds) {
            auto decodeResult = getHypothesis(&beamEnd, unfinishedBeams.hyp.size());

            auto decoderToks = decodeResult.tokens;
            decoderToks.erase(decoderToks.begin()); // initial hyp token
            std::cout << decodeResult.score << " " << tokensToString(decoderToks, 0, N) << std::endl;
        }
    }

    // Usually we take the best beam... but never take rejected beams.
    if (beamEnds[0].score < -100000)
        return nullptr;

    // convert the best beam to a result stringis
    auto decodeResult = getHypothesis(&beamEnds[0], unfinishedBeams.hyp.size());
    std::string result;
    int lastSilence = -1;
    for (int i = 0; i < decodeResult.words.size(); ++i) {
        const auto label = decodeResult.words[i];
        if (label >= 0 && label < dfalm.firstCommandLabel) {
            result = appendSpaced(result, decoderObj->wordDict.getEntry(label), false);
        } else if (label >= dfalm.firstCommandLabel) {
            result = appendSpaced(result, tokensToStringDedup(decodeResult.tokens, lastSilence + 1, i), true);
        }

        const auto token = decodeResult.tokens[i];
        if (token == 0)
            lastSilence = i;
    }

    if (opts->debug)
        std::cout << "  result: " << result << std::endl << std::endl;

    return strdup(result.c_str());
}

bool w2l_make_flattrie(const char *tokens_path, const char *kenlm_model_path, const char *lexicon_path, const char *flattrie_path)
{
    auto tokenDict = Dictionary(tokens_path);
    auto silIdx = tokenDict.getIndex(kSilToken);

    auto lexicon = loadWords(lexicon_path, -1);

    // Adjust the lexicon words to always end in silence
    for (auto &entry : lexicon) {
        for (auto &spelling : entry.second) {
            if (spelling.empty() || spelling.back() != "|")
                spelling.push_back("|");
        }
    }

    auto wordDict = createWordDict(lexicon);
    auto lm = std::make_shared<KenLM>(kenlm_model_path, wordDict);

    // taken from Decode.cpp
    // Build Trie
    std::shared_ptr<Trie> trie = std::make_shared<Trie>(tokenDict.indexSize(), silIdx);
    auto startState = lm->start(false);
    for (auto& it : lexicon) {
        const std::string& word = it.first;
        int usrIdx = wordDict.getIndex(word);
        float score = -1;
        // if (FLAGS_decodertype == "wrd") {
        if (true) {
            LMStatePtr dummyState;
            std::tie(dummyState, score) = lm->score(startState, usrIdx);
        }
        for (auto& tokens : it.second) {
            auto tokensTensor = tkn2Idx(tokens, tokenDict);
            trie->insert(tokensTensor, usrIdx, score);
        }
    }

    // Smearing
    // TODO: smear mode argument?
    SmearingMode smear_mode = SmearingMode::MAX;
    /*
    SmearingMode smear_mode = SmearingMode::NONE;
    if (FLAGS_smearing == "logadd") {
        smear_mode = SmearingMode::LOGADD;
    } else if (FLAGS_smearing == "max") {
        smear_mode = SmearingMode::MAX;
    } else if (FLAGS_smearing != "none") {
        LOG(FATAL) << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
    }
    */
    trie->smear(smear_mode);

    auto flatTrie = std::make_shared<FlatTrie>(toFlatTrie(trie->getRoot()));

    std::ofstream out(flattrie_path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!out.is_open())
        return false;

    size_t byteSize = 4 * flatTrie->storage.size();
    out << byteSize;
    out.write(reinterpret_cast<const char*>(flatTrie->storage.data()), byteSize);
    out.close();
    return out.good();
}

} // extern "C"
