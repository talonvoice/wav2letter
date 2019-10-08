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

std::string tokens = "|'abcdefghijklmnopqrstuvwxyz";
const int TOKENS = 28;
std::vector<uint8_t> charToToken(128);
uint32_t EDGE_INIT[TOKENS] = {0};

enum {
    FLAG_NONE    = 0,
    FLAG_TERM    = 1,
};

enum {
    TOKEN_LMWORD     = 255,
    TOKEN_LMWORD_CTX = 254,
};

struct LM {
    const w2l_dfa_node *dfa;
    const w2l_dfa_node *get(const w2l_dfa_node *base, const int32_t idx) const {
        return reinterpret_cast<const w2l_dfa_node *>(reinterpret_cast<const uint8_t *>(base) + idx);
    }
    int wordStartsBefore = 1000000000;
    float commandScore = 1.5;
    std::vector<int> viterbiToks;
};

struct State {
    const w2l_dfa_node *lex = nullptr;
    uint8_t flags = 0;
    enum : uint8_t {
        FlagWordStarted = 1,
        FlagWordEnded = 2,
        FlagWordLabel = 4,
    };

    // used for making an unordered_set of const State*
    struct Hash {
        const LM &unused;
        size_t operator()(const State *v) const {
            return std::hash<const void*>()(v->lex) ^ v->flags;
        }
    };

    struct Equality {
        const LM &unused;
        int operator()(const State *v1, const State *v2) const {
            return v1->lex == v2->lex && v1->flags == v2->flags;
        }
    };

    // Iterate over labels, calling fn with: the new State, the label index and the lm score
    template <typename Fn>
    void forLabels(const LM &lm, Fn&& fn) const {
        if (flags & FlagWordLabel) {
            fn(*this, reinterpret_cast<const uint8_t*>(lex) - reinterpret_cast<const uint8_t*>(lm.dfa), lm.commandScore);
        }
    }

    // Call finish() on the lm, like for end-of-sentence scoring
    std::pair<State, float> finish(const LM &lm) const {
        return {*this, 0};
    }

    float maxWordScore() const {
        return 0; // could control whether the beam search gets scores before finishing commands
    }

    // Iterate over children of the state, calling fn with:
    // new State, new token index and whether the new state has children
    template <typename Fn>
    bool forChildren(int frame, const LM &lm, Fn&& fn) const {
        if (!(flags & FlagWordStarted) && frame >= lm.wordStartsBefore)
            return true;
        if (flags & FlagWordEnded) {
            fn(State{lex, FlagWordStarted | FlagWordEnded}, lm.viterbiToks[frame], true);
            return false;
        }
        for (int i = 0; i < lex->nEdges; ++i) {
            const auto &edge = lex->edges[i];
            auto nlex = lm.get(lex, edge.offset);
            if (edge.token == TOKEN_LMWORD || edge.token == TOKEN_LMWORD_CTX)
                continue;
            fn(State{nlex, uint8_t(edge.token == 0 ? (FlagWordStarted | FlagWordEnded | FlagWordLabel) : FlagWordStarted)}, edge.token, edge.token != 0);
        }
        return true;
    }

    State &actualize() {
        return *this;
    }
};

} // namespace DFALM

using CommandDecoder = SimpleDecoder<DFALM::LM, DFALM::State>;

// Score adjustment during beam search to reject beams early
// that diverge too much from the best emission-transmission score.
struct CommandViterbiDifferenceRejecter {
    // index i contains the emission-transmission score of up to windowMaxSize
    // previous frames of the viterbiTokens, see precomputeViterbiWindowScores.
    std::vector<float> viterbiWindowScores;

    int windowMaxSize;
    float threshold;
    float *emissions;
    float *transitions;
    int T;

    float extraNewTokenScore(int frame, const CommandDecoder::DecoderState &prevState, int token) const {
        // Stop rejection after decode word end
        if (prevState.lmState.flags & DFALM::State::FlagWordEnded)
            return 0;
        auto refScore = viterbiWindowScores[frame];

        int prevToken = token;
        auto thisState = &prevState;
        float thisScore = emissions[frame * T + token];
        int thisWindow = 1;
        while (thisWindow < windowMaxSize && thisState && frame - thisWindow >= 0) {
            token = thisState->getToken();
            thisScore += emissions[(frame - thisWindow) * T + token] + transitions[prevToken * T + token];
            ++thisWindow;
            prevToken = token;
            thisState = thisState->parent;
        }

        // rejecting based on non-full windows is too unstable
        // only do it after the decode word end
        if (thisWindow < windowMaxSize && token != 0)
            return 0;

        if (thisScore / refScore < threshold) {
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

    auto emissionTransmissionAdjustment = [&transitions, T](const std::vector<int> &tokens, int from, int i, float *emissions) {
        float score = 0;
        if (i > from) {
            score += transitions[tokens[i] * T + tokens[i - 1]];
        } else {
            score += transitions[tokens[i] * T + 0]; // from silence
        }
        score += emissions[i * T + tokens[i]];
        return score;
    };

    auto emissionTransmissionScore = [&emissionTransmissionAdjustment, &transitions, T](const std::vector<int> &tokens, int from, int to, float *emissions) {
        float score = 0.0;
        for (int i = from; i < to; ++i) {
            score += emissionTransmissionAdjustment(tokens, from, i, emissions);
        }
        score += transitions[0 * T + tokens[to - 1]]; // to silence
        return score;
    };

    auto worstEmissionTransmissionWindowFraction = [&emissionVec, &emissionTransmissionAdjustment, &transitions, T](
            const std::vector<int> &tokens1,
            const std::vector<int> &tokens2,
            int from, int to, int window) {
        float score1 = 0.0;
        float score2 = 0.0;
        float worst = INFINITY;
        for (int i = from; i < to; ++i) {
            score1 += emissionTransmissionAdjustment(tokens1, from, i, emissionVec.data());
            score2 += emissionTransmissionAdjustment(tokens2, from, i, emissionVec.data());
            if (i < from + window - 1)
                continue;
            if (worst > score1 / score2)
                worst = score1 / score2;
            score1 -= emissionTransmissionAdjustment(tokens1, from, i - window + 1, emissionVec.data());
            score2 -= emissionTransmissionAdjustment(tokens2, from, i - window + 1, emissionVec.data());
        }
        score1 += transitions[0 * T + tokens1[to - 1]]; // to silence
        score2 += transitions[0 * T + tokens2[to - 1]]; // to silence
        if (worst > score1 / score2)
            worst = score1 / score2;
        return worst;
    };


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
    const auto originalViterbiToks = viterbiToks;
    assert(N == viterbiToks.size());

    // Lets skip decoding if viterbi thinks it's all silence
    bool allSilence = true;
    for (auto t : viterbiToks) {
        if (t != 0)
            allSilence = false;
    }
    if (allSilence)
        return nullptr;

    auto dfalm = DFALM::LM{dfa};
    dfalm.commandScore = opts->command_decoder_opts.wordscore;

    auto commandDecoder = CommandDecoder{
                toW2lDecoderOptions(opts->command_decoder_opts),
                dfalm,
                decoderObj->silIdx,
                decoderObj->wordDict.getIndex(kUnkToken),
                transitions};

    if (opts->debug) {
        std::cout << "detecting in viterbi toks: " << tokensToString(viterbiToks, 0, viterbiToks.size()) << std::endl;
    }

    struct Hyp {
        int endFrame = 0;
        std::string text;
        const w2l_dfa_node *next = nullptr;
        float score = 0;
        std::vector<int> tokens;
        std::vector<int> langDecodeLabels;
        std::vector<int> langDecodeTokens;
    };

    // Do an "outer beamsearch". Hyp contains the unfinished beams.
    // The acceptable ends are collected in ends.
    std::queue<Hyp> hyps;
    std::vector<Hyp> ends;

    hyps.push(Hyp{0, "", dfalm.dfa});

    auto appendSpaced = [&](const std::string &base, const std::string &str, bool command = false) {
        std::string out = base;
        if (!out.empty())
            out += " ";
        if (command)
            out += "@";
        out += str;
        return out;
    };

    auto appendToks = [](const std::vector<int> &base, const std::vector<int> &append, int begin, int end) {
        auto result = base;
        result.insert(result.end(), append.begin() + begin, append.begin() + end);
        return result;
    };

    while (!hyps.empty()) {
        auto hyp = hyps.front();
        hyps.pop();

        DFALM::State commandState;
        commandState.lex = hyp.next;

        bool hypContinues = false;

        int i = hyp.endFrame;
        int segStart = i;

        // Optionally rerun the acoustic model from the current location
        if (i < N && opts->rerun_acoustic) {
            auto nextInput = emissionObj->inputs(af::seq(i * opts->feature_frames_per_output_frame, af::end), af::span, af::span, af::span);
            auto nextEmissions = engineObj->process(nextInput);
            rawEmission(af::span, af::seq(i, af::end), af::span, af::span) = nextEmissions;
            emissionVec = afToVector<float>(rawEmission);
        }

        // Compute the viterbi path starting at token i. This is helpful because often the
        // viterbi path incorrectly merges words and changing the start produces better token
        // assignments.
        // Example: "testtttttask" and when the decoder consumes a part of it (like "test")
        // the upcoming viterbi tokens would be "tttttask". But if the path is recomputed from
        // that offset it'd likely be "|||ttask".
        // Recomputing like this would come at no cost of the temporary data from the viterbi
        // algorithm was kept around and it was run back-to-front.
        if (i < N) {
            auto newViterbiToks =
                afToVector<int>(engineObj->viterbiPath(rawEmission.cols(i, N - 1)));
            for (int j = 0; j < i; ++j)
                newViterbiToks.insert(newViterbiToks.begin(), 0);
            viterbiToks = newViterbiToks;
        }

        while (i < N && viterbiToks[i] == 0)
            ++i;
        int viterbiWordStart = i;
        while (i < N && viterbiToks[i] != 0)
            ++i;
        int viterbiWordEnd = i;
        // it's ok if wordStart == wordEnd, maybe the decoder sees something

        if (opts->debug) {
            std::cout << "  hyp" << std::endl;
            std::cout << "    text so far: " << hyp.text << std::endl;
            std::cout << "    upcoming toks: " << tokensToString(viterbiToks, segStart, viterbiWordEnd) << std::endl;
            std::cout << "    viterbi word: " << tokensToString(viterbiToks, viterbiWordStart, viterbiWordEnd) << std::endl;
        }

        // Find language-mode continuations of hyp.
        const w2l_dfa_node *lang = nullptr;
        bool allowsCommand = false;
        for (int edge = 0; edge < commandState.lex->nEdges; ++edge) {
            const auto &edgeInfo = commandState.lex->edges[edge];
            const w2l_dfa_node *child = dfalm.get(commandState.lex, edgeInfo.offset);
            if (edgeInfo.token != DFALM::TOKEN_LMWORD && edgeInfo.token != DFALM::TOKEN_LMWORD_CTX) {
                allowsCommand = true;
                continue;
            }
            if (segStart == N)
                continue;

            // TODO: We may want different decoding (maybe even acoustic models) depending
            // on whether what follows is a phrase or disjoint words.

            std::vector<int> decodeLabels;
            std::vector<int> decodeTokens;
            if (edgeInfo.token == DFALM::TOKEN_LMWORD_CTX && !hyp.langDecodeLabels.empty()) {
                decodeLabels = hyp.langDecodeLabels;
                decodeTokens = hyp.langDecodeTokens;
            } else {
                KenFlatTrieLM::State langStartState;
                langStartState.kenState = decoderObj->lm->start(0);
                langStartState.lex = decoderObj->flatTrie->getRoot();
                auto languageResult = decoderObj->decoder->normal(emissionVec.data() + segStart * T, N - segStart, T, langStartState);
//                std::cout << "lang decode:" << N - segStart << " " << languageResult.words.size() << " " << languageResult.tokens.size() << std::endl;
//                for (auto j = 0; j < languageResult.words.size(); ++j) {
//                    std::cout << (languageResult.tokens[j] == -1 ? "-" : engineObj->tokenDict.getEntry(languageResult.tokens[j]));
//                }
//                std::cout << std::endl;
//                for (auto j = 0; j < languageResult.words.size(); ++j) {
//                    std::cout << (languageResult.words[j] == -1 ? "-" : "w");
//                }
//                std::cout << std::endl;
                decodeLabels = std::move(languageResult.words);
                decodeTokens = std::move(languageResult.tokens);
                decodeLabels.erase(decodeLabels.begin());
                decodeTokens.erase(decodeTokens.begin());
            }

            // Find the first decode result
            int langWordEnd = 0;
            while (langWordEnd < decodeLabels.size() && decodeLabels[langWordEnd] == -1)
                ++langWordEnd;
            if (langWordEnd == decodeLabels.size())
                continue;

            int decodeLabel = decodeLabels[langWordEnd];
            auto decodeWord = decoderObj->wordDict.getEntry(decodeLabel);
            int endTok = segStart + langWordEnd;

            // Due to negative silweight in the language decoder it's often the case
            // that the decode ends but viterbi toks still run on. Allow consuming
            // repeated tokens until the next silence.
            int vitEnd = endTok;
            while (vitEnd < N && viterbiToks[vitEnd] != 0 && viterbiToks[vitEnd] == viterbiToks[endTok])
                ++vitEnd;
            // If the next token isn't a silence... oops
            if (vitEnd < N && viterbiToks[vitEnd] != 0)
                vitEnd = endTok;

            if (opts->debug) {
                std::cout << "    lang candidate:" << std::endl
                          << "        word: " << decodeWord << std::endl
                          << "        toks: " << tokensToString(decodeTokens, 0, langWordEnd) << std::endl
                          << "       vtoks: " << tokensToString(viterbiToks, segStart, vitEnd) << std::endl;
            }

            // TODO: To support interjecting command words we shouldn't save the actual decode,
            // but the beam search state. We do however want to do the whole phrase decode
            // in one go where possible because the second decoded word can influence the first.
            // A mixed approach where the beam is collapsed when seeing a command word,
            // but the kenlm language state is nevertheless retained seems easiest. And much more
            // complex arrangements are possible.
            auto nextDecodeLabels = std::vector<int>(decodeLabels.begin() + langWordEnd, decodeLabels.end());
            nextDecodeLabels[0] = -1;
            auto nextDecodeTokens = std::vector<int>(decodeTokens.begin() + langWordEnd, decodeTokens.end());

            // Sanity check to guarantee no infinite hyp loops
            if (vitEnd <= segStart)
                continue;

            hyps.push(Hyp{
                vitEnd,
                appendSpaced(hyp.text, decodeWord),
                child,
                hyp.score + emissionTransmissionScore(decodeTokens, 0, langWordEnd, &emissionVec[segStart * T]),
                appendToks(appendToks(hyp.tokens, decodeTokens, 0, langWordEnd), viterbiToks, endTok, vitEnd),
                nextDecodeLabels,
                nextDecodeTokens,
            });
            hypContinues = true;
        }

        // Find command continuations
        [&]() {
            if (!allowsCommand)
                return;
            if (segStart == N)
                return;

            CommandViterbiDifferenceRejecter rejecter;
            rejecter.windowMaxSize = opts->rejection_window_frames;
            rejecter.threshold = opts->early_rejection_threshold;
            rejecter.emissions = emissionVec.data() + segStart * T;
            rejecter.transitions = transitions.data();
            rejecter.T = T;
            rejecter.precomputeViterbiWindowScores(segStart, viterbiToks);

            // in the future we could stop the decode after one word instead of
            // decoding everything
            int decodeLen = N - segStart;
            commandDecoder.lm_.wordStartsBefore = viterbiWordEnd - segStart;
            commandDecoder.lm_.viterbiToks.assign(viterbiToks.begin() + segStart, viterbiToks.end());
            std::vector<CommandDecoder::DecoderState> startStates;
            startStates.emplace_back(commandState, nullptr, 0.0, 0, -1);
            auto beams = commandDecoder.normalAll(emissionVec.data() + segStart * T, decodeLen, T, startStates, rejecter);
            const auto &beamEnds = beams.hyp.back();
            std::set<std::string> seenDecodeWords;
            for (const auto &beamEnd : beamEnds) {
                auto decodeResult = getHypothesis(&beamEnd, beams.hyp.size() - 1);

                // find the recognized word index
                int nextDfaNode = -1;
                for (auto label : decodeResult.words) {
                    if (label != -1) {
                        nextDfaNode = label;
                        break;
                    }
                }
                // reject no-decodes
                if (nextDfaNode == -1)
                    continue;

                auto decoderToks = decodeResult.tokens;
                decoderToks.erase(decoderToks.begin()); // initial hyp token
                std::vector<int> startSil(segStart, 0);
                decoderToks.insert(decoderToks.begin(), startSil.begin(), startSil.end());
                assert(segStart + decodeLen == decoderToks.size());

                int j = 0;
                while (j < decoderToks.size() && decoderToks[j] == 0)
                    ++j;
                int decodeWordStart = j;
                while (j < decoderToks.size() && decoderToks[j] != 0)
                    ++j;
                int decodeWordEnd = j;

                // if the decoder didn't see anything but viterbi saw something, there's no command
                if (decodeWordStart == decodeWordEnd) {
                    continue;
                }

                // Don't look at the same decoded word twice
                const auto decodedWord = tokensToStringDedup(decoderToks, decodeWordStart, decodeWordEnd);
                if (seenDecodeWords.find(decodedWord) != seenDecodeWords.end())
                    continue;
                seenDecodeWords.insert(decodedWord);

                // Compute the start and end of the scored word.
                int scoreWordStart = std::min(viterbiWordStart, decodeWordStart);
                int scoreWordEnd = decodeWordEnd;
                int trailingTokenEnd = decodeWordEnd;
                while (viterbiToks[trailingTokenEnd] == decoderToks[decodeWordEnd - 1] && trailingTokenEnd < N)
                    trailingTokenEnd++;
                if (trailingTokenEnd > decodeWordEnd) {
                    // Often there are trailing tokens.
                    // four has viterbi fourrrrrrr
                    // sometimes it's connected to the next word: fourrrrrrrright
                    if (trailingTokenEnd != N && viterbiToks[trailingTokenEnd] != 0)
                        scoreWordEnd += (trailingTokenEnd - decodeWordEnd) / 2;
                    else
                        scoreWordEnd = trailingTokenEnd;
                    for (int k = decodeWordEnd; k < scoreWordEnd; ++k)
                        decoderToks[k] = decoderToks[decodeWordEnd - 1];
                    decodeWordEnd = scoreWordEnd;
                } else {
                    // extend by one minimum - shoud be at least one frame of silence between words.
                    scoreWordEnd = std::min(decodeWordEnd + 1, N);
                }

                // the criterion for rejecting decodes is the decode-score / viterbi-score
                // where the score is the emission-transmission score
                float windowFrac = worstEmissionTransmissionWindowFraction(decoderToks, viterbiToks, scoreWordStart, scoreWordEnd, opts->rejection_window_frames);
                bool goodCommand = windowFrac > opts->rejection_threshold && nextDfaNode != -1;

                if (opts->debug) {
                    float viterbiScore = emissionTransmissionScore(viterbiToks, scoreWordStart, scoreWordEnd, emissionVec.data());
                    float decoderScore = emissionTransmissionScore(decoderToks, scoreWordStart, scoreWordEnd, emissionVec.data());

                    if (nextDfaNode == -1) {
                        std::cout << "    no command" << std::endl;
                    } else if (goodCommand) {
                        std::cout << "    good command" << std::endl;
                    } else {
                        std::cout << "    rejected command" << std::endl;
                    }
                    std::cout << "       decoder: " << tokensToString(decoderToks, scoreWordStart, scoreWordEnd) << std::endl
                              << "       viterbi: " << tokensToString(viterbiToks, scoreWordStart, scoreWordEnd) << std::endl
                              << "       scores: decoder: " << decoderScore << " viterbi: " << viterbiScore << " fraction: " << decoderScore / viterbiScore << std::endl
                              << "               worst window fraction: " << windowFrac << std::endl;
                }

                if (!goodCommand) {
                    continue; // too many candidates?
                }

                // Sanity check to guarantee no infinite hyp loops
                if (scoreWordEnd <= segStart)
                    break;

                // Tiny adjustment to make it prefer commands if the
                // exact same tokens are valid as language and command.
                const auto commandBonusScore = 1;

                hyps.push(Hyp{
                    decodeWordEnd,
                    appendSpaced(hyp.text, decodedWord, true),
                    dfalm.get(dfalm.dfa, nextDfaNode),
                    hyp.score
                        + emissionTransmissionScore(decoderToks, segStart, decodeWordEnd, emissionVec.data())
                        + commandBonusScore,
                    appendToks(hyp.tokens, decoderToks, segStart, decodeWordEnd),
                });
                hypContinues = true;
            }
        }();

        bool viterbiEndSilence = viterbiWordStart == viterbiWordEnd && viterbiWordEnd == N;

        // If the hyp doesn't continue: accept or reject it as a possible end
        if (!hypContinues || viterbiEndSilence) {
            if (!(hyp.next->flags & DFALM::FLAG_TERM)) {
                if (opts->debug) {
                    std::cout << "    no decode candidates, and not TERM" << std::endl;
                    std::cout << "    discarding: " << hyp.text << std::endl;
                }
            } else if (!viterbiEndSilence) {
                // Otherwise there was something and we discard this hyp
                if (opts->debug) {
                    std::cout << "    no decode candidates, and more follows" << std::endl;
                    std::cout << "    discarding: " << hyp.text << std::endl;
                }
            } else if (!hyp.text.empty()) {
                auto end = hyp;

                // Score the remaining silence
                auto silence = std::vector<int>(N, 0);
                end.score += emissionTransmissionScore(silence, segStart, N, emissionVec.data());

                if (opts->debug) {
                    std::cout << "    accepted end score: " << end.score << std::endl;
                    std::cout << "        text: " << end.text << std::endl;
                }
                ends.push_back(end);
            }
        }
    }

    // Sort ends by score and return the best one
    std::sort(ends.begin(), ends.end(), [](const auto &l, const auto &r) { return l.score > r.score; });
    if (opts->debug) {
        for (const auto &end : ends) {
            // Could do rejection here.
            //auto rej = worstEmissionTransmissionWindowFraction(end.tokens, originalViterbiToks, 0, end.tokens.size(), opts->rejection_window_frames);
            //if (rej < 0.6)
            //    continue;
            std::cout << "  possible result: " << end.score << " " << end.text << std::endl;
        }
    }
    for (const auto &end : ends) {
        if (opts->debug)
            std::cout << "  result: " << end.text << std::endl << std::endl;
        return strdup(end.text.c_str());
    }
    return nullptr;
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
