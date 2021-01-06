#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <unordered_set>

#include "common/Transforms.h"
#include "common/Utils.h"
#include "libraries/common/Dictionary.h"
#include "libraries/decoder/Utils.h"
#include "libraries/decoder/Trie.h"
#include "libraries/lm/KenLM.h"
#include "libraries/common/WordUtils.h"

// for viterbi path
#include "libraries/criterion/cpu/ViterbiPath.h"

using namespace w2l;

#include "w2l_decode.h"
#include "decode_core.cpp"

namespace w2l {
// from common/Utils.h (which includes flashlight, so we don't include it)
std::string join(const std::string& delim, const std::vector<std::string>& vec);
}

class GreedyDecoder {
public:
    static void decodeASG(w2l_emission *emission, float *transitions, int *path_out) {
        int T = emission->n_frames;
        int N = emission->n_tokens;
        float *emissions = emission->matrix;
        std::vector<uint8_t> workspace(w2l::cpu::ViterbiPath<float>::getWorkspaceSize(1, T, N));
        w2l::cpu::ViterbiPath<float>::compute(
            1, // B
            T,
            N,
            emissions,
            transitions,
            path_out,
            workspace.data());
    }

    static void decodeCTC(w2l_emission *emission, int *path_out) {
        int T = emission->n_frames;
        int N = emission->n_tokens;
        float *emissions = emission->matrix;
        // CTC viterbi is just argmax
        for (int t = 0; t < T; t++) {
            auto it = std::max_element(emissions, emissions + N);
            path_out[t] = std::distance(emissions, it);
            emissions += N;
        }
    }
};

DecoderOptions toW2lDecoderOptions(const w2l_decode_options &opts) {
    CriterionType criterionType;
    if (std::string(opts.criterion) == kCtcCriterion) {
        criterionType = CriterionType::CTC;
    } else if (std::string(opts.criterion) == kAsgCriterion) {
        criterionType = CriterionType::ASG;
    } else {
        std::cerr << "[Decoder] Invalid criterion type: " << opts.criterion << std::endl;
        abort();
    }
    return DecoderOptions(
                opts.beamsize,
                opts.beamsizetoken,
                opts.beamthresh,
                opts.lmweight,
                opts.wordscore, // lexiconscore
                opts.unkweight, // unkscore
                opts.silweight, // silscore
                0,              // eosscore
                opts.logadd,
                criterionType);
}

std::vector<std::string> loadWordList(const char *path) {
    std::vector<std::string> result;
    result.reserve(1000000);
    result.push_back("<unk>");

    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        auto sep = std::min(line.find("\t"), line.find(" "));
        auto word = line.substr(0, sep);
        // handle duplicate words
        if (result.size() == 0 || word != result.back()) {
            result.push_back(word);
        }
    }

    return result;
}

static int getSilIdx(Dictionary &tokenDict) {
    if (tokenDict.contains(kSilToken)) {
        return tokenDict.getIndex(kSilToken);
    } else if (tokenDict.contains("_")) {
        return tokenDict.getIndex("_");
    } else if (tokenDict.contains("|")) {
        return tokenDict.getIndex("|");
    }
    return 0;
}

class PublicDecoder {
public:
    PublicDecoder(const char *tokens, const char *languageModelPath, const char *lexiconPath, const w2l_decode_options *opts) {
        this->lexiconPath = lexiconPath;
        this->setOptions(opts);
        auto tokenStream = std::istringstream(tokens);
        tokenDict = Dictionary(tokenStream);
        // TODO: ensure that resulting tokenDict.indexSize() > 0?
        if (tokenDict.indexSize() <= 0) {
            std::cerr << "[Decoder] tokenDict.indexSize() <= 0" << std::endl;
            abort();
        }
        if (decoderOpt.criterionType == CriterionType::CTC &&
                tokenDict.indexSize() > 0 &&
                tokenDict.getEntry(tokenDict.indexSize() - 1) != kBlankToken) {
            tokenDict.addEntry(kBlankToken);
        }

        globalTokens = &tokenDict;
        silIdx = getSilIdx(tokenDict);
        if (tokenDict.contains(kBlankToken)) {
            blankIdx = tokenDict.getIndex(kBlankToken);
        } else {
            blankIdx = -1;
        }
        if (decoderOpt.criterionType == CriterionType::ASG) {
            blankIdx = silIdx;
        }

        wordList = loadWordList(lexiconPath);
        lm = std::make_shared<KenLM>(languageModelPath, wordList);
    }
    ~PublicDecoder() {}

    // FLATTRIE HEADER:
    //  0: 'FLAT' (magic)
    //  4: 4-byte version (1)
    //  8: 32-byte  src file hash slot (written as zeroes, up to the caller to fill/check it)
    // 40: 32-byte trie file hash slot (written as zeroes, up to the caller to fill/check it)
    // 72: 8-byte data size
    // 80: data...
    bool loadTrie(const char *triePath) {
        // Load the trie
        std::ifstream f(triePath, std::ios::binary | std::ios::in);
        if (!f) {
            return false;
        }
        char     magic[4];
        char     srcHash[32];
        char     trieHash[32];
        uint32_t version  = 0;
        size_t   byteSize = 0;

        f.read(         magic,    4);
        f.read((char *)&version,  sizeof(version));
        f.read(         srcHash,  sizeof(srcHash));
        f.read(         trieHash, sizeof(trieHash));
        f.read((char *)&byteSize, sizeof(byteSize));
        if (memcmp(magic, "FLAT", 4) != 0) {
            return false;
        }
        if (version != 1) {
            return false;
        }
        if ((byteSize % 4) != 0) {
            return false;
        }
        flatTrie = std::make_shared<FlatTrie>();
        flatTrie->storage.resize(byteSize / 4);
        f.read(reinterpret_cast<char *>(flatTrie->storage.data()), byteSize);

        // the root maxScore should be 0 during search and it's more convenient to set here
        const_cast<FlatTrieNode *>(flatTrie->getRoot())->maxScore = 0;
        return f.good();
    }

    bool makeTrie(const char *triePath) {
        auto lexicon = loadWords(lexiconPath, -1);
        Dictionary wordDict;
        for (const auto& it : wordList) {
            wordDict.addEntry(it);
        }
        wordDict.setDefaultIndex(wordDict.getIndex(kUnkToken));

        // taken from Decode.cpp
        // Build Trie
        Trie trie(tokenDict.indexSize(), silIdx);
        auto startState = lm->start(false);
        for (auto& it : lexicon) {
            const std::string& word = it.first;
            int usrIdx = wordDict.getIndex(word);
            float score = -1;
            // if (FLAGS_decodertype == "wrd")
            if (true) {
                LMStatePtr dummyState;
                std::tie(dummyState, score) = lm->score(startState, usrIdx);
            }
            for (auto& tokens : it.second) {
                auto tokensTensor = tkn2Idx(tokens, tokenDict, false /* replabel */ );
                trie.insert(tokensTensor, usrIdx, score);
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
        trie.smear(smear_mode);

        auto flatTrie = toFlatTrie(trie.getRoot());
        std::ofstream out(triePath, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out.is_open())
            return false;

        // header is described above the loadTrie() method
        const char *magic = "FLAT";
        uint32_t version = 1;
        char zeroHash[32];
        memset(zeroHash, 0, 32);
        uint64_t byteSize = 4 * flatTrie.storage.size();

        out.write(          magic,    4);
        out.write((char *) &version,  sizeof(version));
        out.write(          zeroHash, sizeof(zeroHash));
        out.write(          zeroHash, sizeof(zeroHash));
        out.write((char *) &byteSize, sizeof(byteSize));
        out.write((char *) (flatTrie.storage.data()), byteSize);
        out.close();
        return out.good();
    }

    void setOptions(const w2l_decode_options *opts) {
        // safely retain external opts by copying transitions array and criterion string
        this->opts = *opts;
        this->opts.transitions = nullptr;
        this->transitions.resize(opts->transitions_len);
        if (opts->transitions != nullptr && opts->transitions_len > 0) {
            std::copy(opts->transitions, opts->transitions + opts->transitions_len, this->transitions.begin());
        }
        if (opts->criterion == std::string("asg")) {
            this->opts.criterion = "asg";
        } else if (opts->criterion == std::string("ctc")) {
            this->opts.criterion = "ctc";
        } else if (opts->criterion == std::string("s2s")) {
            this->opts.criterion = "s2s";
        } else {
            this->opts.criterion = "";
        }
        decoderOpt = toW2lDecoderOptions(*opts);
    }

    DecodeResult decode(w2l_emission *emission) {
        KenFlatTrieLM::State startState;
        startState.lex = flatTrie->getRoot();
        startState.kenState = lm->start(0);

        KenFlatTrieLM::LM lmWrap;
        lmWrap.ken = lm;
        lmWrap.trie = flatTrie;
        auto decoder = SimpleDecoder<KenFlatTrieLM::LM, KenFlatTrieLM::State>{
            lmWrap,
            silIdx,
            blankIdx,
            unkLabel,
            transitions};
        return decoder.normal(decoderOpt, emission, startState);
        //return decoder.groupThreading(emissionVec.data(), T, N);
    }

    void decodeGreedy(w2l_emission *emission, int *path) {
        if (decoderOpt.criterionType == CriterionType::CTC) {
            GreedyDecoder::decodeCTC(emission, path);
        } else if (decoderOpt.criterionType == CriterionType::ASG) {
            int64_t tokens_squared = emission->n_tokens * emission->n_tokens;
            if (transitions.size() != tokens_squared) {
                std::cerr << "[Decoder] transitions.size() {" << transitions.size() << "} != emission->n_tokens ** 2 {" << tokens_squared << "}" << std::endl;
                abort();
            }
            GreedyDecoder::decodeASG(emission, &transitions[0], path);
        } else {
            std::cerr << "[Decoder] Unknown criterion enum: " << (int)decoderOpt.criterionType << std::endl;
            abort();
        }
    }

    char *decodeDFA(w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size);

    char *resultWords(const DecodeResult &result) {
        auto rawWordPrediction = validateIdx(result.words, unkLabel);
        std::vector<std::string> wordPrediction;
        for (auto wrdIdx : rawWordPrediction) {
            wordPrediction.push_back(wordList[wrdIdx]);
        }
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
    std::string lexiconPath;
    std::vector<std::string> wordList;
    Dictionary tokenDict;
    DecoderOptions decoderOpt;
    int silIdx;
    int blankIdx;
    int unkLabel = 0;

    std::vector<float> transitions;
    w2l_decode_options opts = {};
};

namespace DFALM {

// flags on dfa nodes
enum {
    FLAG_NONE    = 0,
    FLAG_TERM    = 1,
};

// special token values on dfa edges
enum {
    TOKEN_LMWORD      = 0xffff,
    TOKEN_LMWORD_CTX  = 0xfffe,
    TOKEN_SKIP        = 0xfffd,
};

struct LM {
    LMPtr ken;
    LMStatePtr kenStart;
    FlatTriePtr trie;
    const w2l_dfa_node *dfa;
    int silToken;
    bool charLM;
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
            return v1->grammarLex == v2->grammarLex
                && v1->dictLex == v2->dictLex
                && (v1->kenState == v2->kenState
                    || (v1->kenState && v2->kenState && v1->kenState == v2->kenState));
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
        if (lm.charLM && wordEnd) {
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
    bool forChildren(int frame, std::unordered_set<int> &indices, const LM &lm, Fn&& fn) const {
        // If a dictionary word was started only consider its trie children.
        if (dictLex) {
            const auto n = dictLex->nChildren;
            for (int i = 0; i < n; ++i) {
                auto nlex = dictLex->child(i);
                if (indices.find(nlex->idx) != indices.end()) {
                    fn(State{grammarLex, nlex, kenState}, nlex->idx, nlex->nChildren > 0);
                }
            }
            return true;
        }

        // Otherwise look at the grammar dfa
        std::vector<const w2l_dfa_node *> queue = {grammarLex};
        while (queue.size() > 0) {
            auto dfaLex = queue.back();
            queue.pop_back();

            for (int i = 0; i < dfaLex->nEdges; ++i) {
                const auto &edge = dfaLex->edges[i];
                auto nlex = lm.get(dfaLex, edge.offset);

                // For dictionary edges start exploring the trie
                if (edge.token == TOKEN_LMWORD || edge.token == TOKEN_LMWORD_CTX) {
                    auto nextKenState = edge.token == TOKEN_LMWORD_CTX ? kenState : nullptr;
                    if (!nextKenState)
                        nextKenState = lm.kenStart;
                    auto dictRoot = lm.trie->getRoot();
                    const auto n = dictRoot->nChildren;
                    for (int i = 0; i < n; ++i) {
                        auto nDictLex = dictRoot->child(i);
                        if (indices.find(nDictLex->idx) != indices.end()) {
                            fn(State{nlex, nDictLex, nextKenState}, nDictLex->idx, nDictLex->nChildren > 0);
                        }
                    }
                } else if (edge.token == TOKEN_SKIP) {
                    // std::cout << "skip token, queueing up a new node with " << nlex->nEdges << " edges\n";
                    queue.push_back(nlex);
                } else if (indices.find(edge.token) != indices.end()) {
                    fn(State{nlex, nullptr, nullptr, edge.token == lm.silToken}, edge.token, true);
                }
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
    int silIdx;
    float threshold;
    w2l_emission *emission;
    float *transitions;

    float extraNewTokenScore(int frame, const CombinedDecoder::DecoderState &prevState, int token) const {
        const int T = emission->n_frames;
        const int N = emission->n_tokens;
        auto refScore = viterbiWindowScores[frame];

        bool allSilence = token == silIdx;
        int prevToken = token;
        auto thisState = &prevState;
        float thisScore = emission->matrix[frame * N + token];
        int thisWindow = 1;
        while (thisWindow < windowMaxSize && thisState && frame - thisWindow >= 0) {
            token = thisState->getToken();
            if (token != 0)
                allSilence = false;
            thisScore += emission->matrix[(frame - thisWindow) * N + token];
            if (transitions) {
                thisScore += transitions[prevToken * N + token];
            }
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
        const int T = emission->n_frames;
        const int N = emission->n_tokens;
        float score = 0;
        for (int j = segStart; j < T; ++j) {
            score += emission->matrix[(j - segStart) * N + viterbiToks[j]];
            if (j != segStart && transitions)
                score += transitions[viterbiToks[j] * N + viterbiToks[j - 1]];
            viterbiWindowScores.push_back(score);
            if (j - segStart < windowMaxSize - 1)
                continue;
            auto r = j - (windowMaxSize - 1);
            score -= emission->matrix[(r - segStart) * N + viterbiToks[r]];
            if (r != segStart && transitions)
                score -= transitions[viterbiToks[r] * N + viterbiToks[r - 1]];
        }
    }
};

char *PublicDecoder::decodeDFA(w2l_emission *emission, w2l_dfa_node *dfa, size_t dfa_size) {
    auto tokensToString = [this](const std::vector<int> &tokens, int from, int to) {
        std::string out;
        for (int i = from; i < to; ++i) {
            out.append(tokenDict.getEntry(tokens[i]));
        }
        return out;
    };
    auto tokensToStringDedup = [this](const std::vector<int> &tokens, int from, int to) {
        std::ostringstream ostr;
        int tok = -1;
        bool lastBlank = false;
        for (int i = from; i < to; ++i) {
            if (tok == tokens[i])
                continue;
            tok = tokens[i];
            if (tok >= 0 && tok != blankIdx) {
                std::string s = tokenDict.getEntry(tok);
                if (!s.empty() && s[0] == '_') {
                    if (ostr.tellp() > 0) {
                        ostr << " ";
                    }
                    s = s.substr(1);
                }
                ostr << s;
            }
            lastBlank = (tok == blankIdx);
        }
        return ostr.str();
    };

    // TODO: we already do viterbi from Talon, so allow passing in a cached viterbi path?
    std::vector<int> viterbiToks(emission->n_frames);
    decodeGreedy(emission, &viterbiToks[0]);

    // Lets skip decoding if viterbi thinks it's all silence
    bool allSilence = true;
    for (auto t : viterbiToks) {
        if (t != silIdx && t != blankIdx) {
            allSilence = false;
            break;
        }
    }
    if (allSilence)
        return nullptr;

    auto dfalm = DFALM::LM{lm, lm->start(0), flatTrie, dfa, silIdx, decoderOpt.criterionType == CriterionType::ASG};
    dfalm.commandScore = opts.command_score;
    dfalm.firstCommandLabel = wordList.size();

    auto commandDecoder = CombinedDecoder{
        dfalm,
        silIdx,
        blankIdx,
        unkLabel,
        transitions};

    if (opts.debug) {
        std::cerr << "detecting in viterbi toks: " << tokensToString(viterbiToks, 0, viterbiToks.size()) << std::endl;
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
    rejecter.windowMaxSize = opts.rejection_window_frames;
    if (rejecter.windowMaxSize <= 0) {
        rejecter.windowMaxSize = 1;
    }
    rejecter.threshold = opts.rejection_threshold;
    rejecter.emission = emission;
    rejecter.silIdx = silIdx;
    if (transitions.size() == 0) {
        rejecter.transitions = NULL;
    } else {
        rejecter.transitions = transitions.data();
    }
    rejecter.precomputeViterbiWindowScores(0, viterbiToks);
    rejecter.viterbiToks = viterbiToks;

    DFALM::State commandState;
    commandState.grammarLex = dfalm.dfa;

    // in the future we could stop the decode after one word instead of
    // decoding everything
    std::vector<CombinedDecoder::DecoderState> startStates;
    startStates.emplace_back(commandState, nullptr, 0.0, silIdx, -1);

    auto unfinishedBeams = [&]() {
        const auto parallelBeamsearch = false;
        if (!parallelBeamsearch)
            return commandDecoder.normalAll(decoderOpt, emission, startStates, rejecter);

        int nThreads = 4;
        int stepsPerFanout = 5;
        int threadBeamSize = decoderOpt.beamSize / nThreads;
        return commandDecoder.groupThreading(decoderOpt, emission, startStates, rejecter, nThreads, stepsPerFanout, threadBeamSize);
    }();

    // Finishing kills beams that end in the middle of a word, or
    // in a grammar state that isn't TERM
    std::vector<CombinedDecoder::DecoderState> beamEnds;
    beamSearchFinish(beamEnds, unfinishedBeams.hyp.back(), dfalm, decoderOpt);

    if (beamEnds.empty())
        return nullptr;

    if (opts.debug) {
        for (const auto &beamEnd : beamEnds) {
            auto decodeResult = _getHypothesis(&beamEnd, emission->n_frames + 1);
            auto decoderToks = decodeResult.tokens;
            std::cerr << decodeResult.score << " " << tokensToString(decoderToks, 1, decoderToks.size() - 1) << std::endl;
        }
    }

    // Usually we take the best beam... but never take rejected beams.
    if (beamEnds[0].score < -100000)
        return nullptr;

    // convert the best beam to a result string
    auto decodeResult = _getHypothesis(&beamEnds[0], unfinishedBeams.hyp.size());
    std::string result;

    if (decoderOpt.criterionType == CriterionType::CTC) {
        result = tokensToStringDedup(decodeResult.tokens, 1, decodeResult.tokens.size() - 1);
    } else {
        int lastSilence = -1;
        for (int i = 0; i < decodeResult.words.size() - 1; ++i) {
            const auto label = decodeResult.words[i];
            if (label >= 0 && label < dfalm.firstCommandLabel) {
                result = appendSpaced(result, wordList[label], false);
            } else if (label >= dfalm.firstCommandLabel) {
                result = appendSpaced(result, tokensToStringDedup(decodeResult.tokens, lastSilence + 1, i), true);
            }
            const auto token = decodeResult.tokens[i];
            if (token == silIdx)
                lastSilence = i;
        }
    }
    if (opts.debug)
        std::cerr << "  result: " << result << std::endl << std::endl;

    // TODO: return DecodeResult instead?
    return strdup(result.c_str());
}
