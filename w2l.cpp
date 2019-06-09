#include <stdlib.h>
#include <string>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lDataset.h"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"
#include "decoder/Decoder.hpp"
#include "decoder/KenLM.hpp"
#include "decoder/Trie.hpp"

using namespace w2l;

class EngineBase {
public:
    int numClasses;
    std::unordered_map<std::string, std::string> config;
    std::shared_ptr<fl::Module> network;
    std::shared_ptr<SequenceCriterion> criterion;
    std::string criterionType;
    Dictionary tokenDict;
};

class Emission {
public:
    Emission(EngineBase *engine, fl::Variable emission) {
        this->engine = engine;
        this->emission = emission;
    }
    ~Emission() {}

    char *text() {
        auto viterbiPath = afToVector<int>(engine->criterion->viterbiPath(emission.array()));
        if (engine->criterionType == kCtcCriterion || engine->criterionType == kAsgCriterion) {
            uniq(viterbiPath);
        }
        remapLabels(viterbiPath, engine->tokenDict);
        auto letters = tensor2letters(viterbiPath, engine->tokenDict);
        if (letters.size() > 0) {
            std::string str = tensor2letters(viterbiPath, engine->tokenDict);
            return strdup(str.c_str());
        }
        return strdup("");
    }

    EngineBase *engine;
    fl::Variable emission;
};

class Engine : public EngineBase {
public:
    Engine(const char *acousticModelPath, const char *tokensPath) {
        // TODO: set criterionType "correctly"
        W2lSerializer::load(acousticModelPath, config, network, criterion);
        auto flags = config.find(kGflags);
        // loading flags globally like this is gross, only way to work around it will be parameterizing everything about wav2letter better
        gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

        criterionType = FLAGS_criterion;
        network->eval();
        criterion->eval();

        tokenDict = createTokenDict(tokensPath);
        numClasses = tokenDict.indexSize();
    }
    ~Engine() {}

    Emission *process(float *samples, size_t sample_count) {
        struct W2lLoaderData data = {};
        std::copy(samples, samples + sample_count, std::back_inserter(data.input));

        auto feat = featurize({data}, {});
        auto result = af::array(feat.inputDims, feat.input.data());
        auto rawEmission = network->forward({fl::input(result)}).front();
        return new Emission(this, rawEmission);
    }
};

class WrapDecoder {
public:
    WrapDecoder(Engine *engine, const char *languageModelPath, const char *lexiconPath) {
        std::cout << 1 << std::endl;
        auto lexicon = loadWords(lexiconPath, -1);
        wordDict = createWordDict(lexicon);
        std::cout << 2 << std::endl;

        int silIdx = engine->tokenDict.getIndex(kSilToken);
        int blankIdx = engine->criterionType == kCtcCriterion ? engine->tokenDict.getIndex(kBlankToken) : -1;
        int unkIdx = lm->index(kUnkToken);
        auto trie = std::make_shared<Trie>(engine->tokenDict.indexSize(), silIdx);
        auto start_state = lm->start(false);
        for (auto& it : lexicon) {
            std::string word = it.first;
            int lmIdx = lm->index(word);
            if (lmIdx == unkIdx) { // We don't insert unknown words
                continue;
            }
            float score;
            auto dummyState = lm->score(start_state, lmIdx, score);
            for (auto& tokens : it.second) {
                auto tokensTensor = tokens2Tensor(tokens, engine->tokenDict);
                trie->insert(
                        tokensTensor,
                        std::make_shared<TrieLabel>(lmIdx, wordDict.getIndex(word)),
                        score);
            }
        }
        std::cout << 3 << std::endl;

        SmearingMode smear_mode = SmearingMode::LOGADD;
        // TODO: smear mode argument?
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
        std::cout << 4 << std::endl;

        auto unk = std::make_shared<TrieLabel>(unkIdx, wordDict.getIndex(kUnkToken));
        decoder = std::make_shared<Decoder>(trie, lm, silIdx, blankIdx, unk);

        std::cout << 5 << std::endl;
        lm = std::make_shared<KenLM>(languageModelPath);

        std::cout << 6 << std::endl;
        ModelType modelType = ModelType::ASG;
        if (engine->criterionType == kCtcCriterion) {
            modelType = ModelType::CTC;
        } else if (engine->criterionType != kAsgCriterion) {
            // FIXME:
            LOG(FATAL) << "[Decoder] Invalid model type: " << engine->criterionType;
        }
        std::cout << 7 << std::endl;

        // FIXME, don't use global flags
        decoderOpt = DecoderOptions(
            FLAGS_beamsize,
            static_cast<float>(FLAGS_beamscore),
            static_cast<float>(FLAGS_lmweight),
            static_cast<float>(FLAGS_wordscore),
            static_cast<float>(FLAGS_unkweight),
            FLAGS_forceendsil,
            FLAGS_logadd,
            static_cast<float>(FLAGS_silweight),
            modelType);
        std::cout << 8 << std::endl;
    }
    ~WrapDecoder() {}

    char *decode(Emission *emission) {
        auto transition = afToVector<float>(emission->engine->criterion->param(0).array());
        auto rawEmission = emission->emission;
        auto emissionVec = afToVector<float>(rawEmission);
        int N = rawEmission.dims(0);
        int T = rawEmission.dims(1);

        std::vector<float> score;
        std::vector<std::vector<int>> wordPredictions;
        std::vector<std::vector<int>> letterPredictions;
        std::tie(score, wordPredictions, letterPredictions) = decoder->decode(
            decoderOpt, &transition[0], emissionVec.data(), T, N);
        auto wordPrediction = wordPredictions[0];
        auto words = tensor2words(wordPrediction, wordDict);
        return strdup(words.c_str());
    }

    std::shared_ptr<KenLM> lm;
    std::shared_ptr<Decoder> decoder;
    Dictionary wordDict;
    DecoderOptions decoderOpt;
};

extern "C" {

#include "w2l.h"

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_emission w2l_emission;

w2l_engine *w2l_engine_new(const char *acoustic_model_path, const char *tokens_path) {
    // TODO: what other engine config do I need?
    auto engine = new Engine(acoustic_model_path, tokens_path);
    return reinterpret_cast<w2l_engine *>(engine);
}

w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count) {
    auto emission = reinterpret_cast<Engine *>(engine)->process(samples, sample_count);
    return reinterpret_cast<w2l_emission *>(emission);
}

void w2l_engine_free(w2l_engine *engine) {
    if (engine)
        delete reinterpret_cast<Engine *>(engine);
}

char *w2l_emission_text(w2l_emission *emission) {
    // TODO: I think w2l_emission needs a pointer to the criterion to do viterbiPath
    //       I could just use a shared_ptr to just the criterion and not point emission -> engine
    //       so I'm not passing a raw shared_ptr back from C the api
    // TODO: do a viterbiPath here
    return reinterpret_cast<Emission *>(emission)->text();
}

void w2l_emission_free(w2l_emission *emission) {
    if (emission)
        delete reinterpret_cast<Emission *>(emission);
}

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path) {
    // TODO: what other config? beam size? smearing? lm weight?
    auto decoder = new WrapDecoder(reinterpret_cast<Engine *>(engine), kenlm_model_path, lexicon_path);
    return reinterpret_cast<w2l_decoder *>(decoder);
}

char *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission) {
    return reinterpret_cast<WrapDecoder *>(decoder)->decode(reinterpret_cast<Emission *>(emission));
}

void w2l_decoder_free(w2l_decoder *decoder) {
    if (decoder)
        delete reinterpret_cast<WrapDecoder *>(decoder);
}

} // extern "C"
