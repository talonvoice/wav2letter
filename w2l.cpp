#include <iostream>
#include <stdlib.h>
#include <string>
#include <typeinfo>

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
#include "decoder/Decoder.h"
#include "decoder/Trie.h"
#include "decoder/WordLMDecoder.h"
#include "decoder/TokenLMDecoder.h"
#include "lm/KenLM.h"

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
        auto tokenPrediction =
            afToVector<int>(engine->criterion->viterbiPath(emission.array()));
        auto letters = tknPrediction2Ltr(tokenPrediction, engine->tokenDict);
        if (letters.size() > 0) {
            std::ostringstream ss;
            for (auto s : letters) ss << s;
            return strdup(ss.str().c_str());
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

        tokenDict = Dictionary(tokensPath);
        numClasses = tokenDict.indexSize();
    }
    ~Engine() {}

    Emission *process(float *samples, size_t sample_count) {
        struct W2lLoaderData data = {};
        std::copy(samples, samples + sample_count, std::back_inserter(data.input));

        auto feat = featurize({data}, {});
        auto input = af::array(feat.inputDims, feat.input.data());
        auto rawEmission = network->forward({fl::input(input)}).front();
        return new Emission(this, rawEmission);
    }

    bool exportModel(const char *path) {
        std::ofstream outfile;
        outfile.open(path, std::ios::out | std::ios::binary);
        if (!outfile.is_open()) {
                std::cout << "[w2lapi] error, could not open file '" << path << "' (aborting export)" << std::endl;
            return false;
        }

        auto seq = dynamic_cast<fl::Sequential *>(network.get());
        for (auto &module : seq->modules()) {
            if (!exportLayer(outfile, module.get())) {
                std::cout << "[w2lapi] aborting export" << std::endl;
                return false;
            }
        }
        return true;
    }

private:
    std::tuple<int, int> splitOn(std::string s, std::string on) {
        auto split = s.find(on);
        auto first = s.substr(0, split);
        auto second = s.substr(split + on.size());
        // std::cout << "string [" << s << "] on [" << on << "] first " << first << " second " << second << std::endl;
        return {std::stoi(first), std::stoi(second)};
    }

    std::string findParens(std::string s) {
        auto start = s.find('(');
        auto end = s.find(')', start);
        auto sp = s.substr(start + 1, end - start - 1);
        // std::cout << "string split [" << s << "] " << start << " " << end << " [" << sp << "]" << std::endl;
        return sp;
    }

    void exportParams(std::ofstream& f, fl::Variable params) {
        auto array = afToVector<float>(params.array());
        for (auto& p : array) {
            f << std::hex << (uint32_t&)p;
            if (&p != &array.back()) {
                f << " ";
            }
        }
        f << std::dec;
    }

    bool exportLayer(std::ofstream& f, fl::Module *module) {
        auto pretty = module->prettyString();
        auto type = pretty.substr(0, pretty.find(' '));
        std::cout << "[w2lapi] exporting: " << pretty << std::endl;
        if (type == "WeightNorm") {
            auto wn = dynamic_cast<fl::WeightNorm *>(module);
            auto lastParam = pretty.rfind(",") + 2;
            auto dim = pretty.substr(lastParam, pretty.size() - lastParam - 1);
            f << "WN " << dim << " ";
            exportLayer(f, wn->module().get());
        } else if (type == "View") {
            auto ratio = findParens(pretty);
            f << "V " << findParens(pretty) << "\n";
        } else if (type == "Dropout") {
            f << "DO " << findParens(pretty) << "\n";
        } else if (type == "Reorder") {
            auto dims = findParens(pretty);
            std::replace(dims.begin(), dims.end(), ',', ' ');
            f << "RO " << dims << "\n";
        } else if (type == "GatedLinearUnit") {
            f << "GLU " << findParens(pretty) << "\n";
        } else if (type == "Conv2D") {
            // Conv2D (234->514, 23x1, 1,1, 0,0, 1, 1) (with bias)
            auto parens = findParens(pretty);
            bool bias = pretty.find("with bias") >= 0;
            int inputs, outputs, szX, szY, padX, padY, strideX, strideY, dilateX, dilateY;
            // TODO: I could get some of these from the params' dims instead of string parsing...

            auto comma1 = parens.find(',') + 1;
            std::tie(inputs, outputs) = splitOn(parens.substr(0, comma1), "->");

            auto comma2 = parens.find(',', comma1) + 1;
            std::tie(szX, szY) = splitOn(parens.substr(comma1, comma2 - comma1 - 1), "x");

            auto comma4 = parens.find(',', parens.find(',', comma2) + 1) + 1;
            std::tie(strideX, strideY) = splitOn(parens.substr(comma2, comma4 - comma2 - 1), ",");

            auto comma6 = parens.find(',', parens.find(',', comma4) + 1) + 1;
            std::tie(padX, padY) = splitOn(parens.substr(comma4, comma6 - comma4 - 1), ",");

            auto comma8 = parens.find(',', parens.find(',', comma6) + 1) + 1;
            std::tie(dilateX, dilateY) = splitOn(parens.substr(comma6, comma8 - comma6 - 1), ",");

            // FIXME we're ignoring everything after padX because I don't know the actual spec
            // string split [Conv2D (40->200, 13x1, 1,1, 170,0, 1, 1) (with bias)] 7 39 [40->200, 13x1, 1,1, 170,0, 1, 1]
            // fl::Conv2D C2 [inputChannels] [outputChannels] [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding <OPTIONAL>] [yPadding <OPTIONAL>] [xDilation <OPTIONAL>] [yDilation <OPTIONAL>]
            f << "C " << inputs << " " << outputs << " " << szX << " " << szY << " " << padX << " | ";
            exportParams(f, module->param(0));
            if (bias) {
                f << " | ";
                exportParams(f, module->param(1));
            }
            f << "\n";
        } else if (type == "Linear") {
            int inputs, outputs;
            std::tie(inputs, outputs) = splitOn(findParens(pretty), "->");
            f << "L " << inputs << " " << outputs << " | ";
            exportParams(f, module->param(0));
            f << "\n";
        } else {
            // TODO: also write error to the file?
            std::cout << "[w2lapi] error, unknown layer type: " << type << std::endl;
            return false;
        }
        return true;
    }
};

class WrapDecoder {
public:
    WrapDecoder(Engine *engine, const char *languageModelPath, const char *lexiconPath) {
        auto lexicon = loadWords(lexiconPath, -1);
        wordDict = createWordDict(lexicon);
        lm = std::make_shared<KenLM>(languageModelPath, wordDict);

        // taken from Decode.cpp
        // Build Trie
        int silIdx = engine->tokenDict.getIndex(kSilToken);
        int blankIdx = engine->criterionType == kCtcCriterion ? engine->tokenDict.getIndex(kBlankToken) : -1;
        std::shared_ptr<Trie> trie = std::make_shared<Trie>(engine->tokenDict.indexSize(), silIdx);
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
                auto tokensTensor = tkn2Idx(tokens, engine->tokenDict);
                trie->insert(tokensTensor, usrIdx, score);
            }
        }

        // Smearing
        // TODO: smear mode argument?
        SmearingMode smear_mode = SmearingMode::LOGADD;
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

        CriterionType criterionType = CriterionType::ASG;
        if (engine->criterionType == kCtcCriterion) {
            criterionType = CriterionType::CTC;
        } else if (engine->criterionType != kAsgCriterion) {
            // FIXME:
            LOG(FATAL) << "[Decoder] Invalid model type: " << engine->criterionType;
        }
        // FIXME, don't use global flags
        DecoderOptions decoderOpt(
            FLAGS_beamsize,
            static_cast<float>(FLAGS_beamthreshold),
            static_cast<float>(FLAGS_lmweight),
            static_cast<float>(FLAGS_wordscore),
            static_cast<float>(FLAGS_unkweight),
            FLAGS_logadd,
            static_cast<float>(FLAGS_silweight),
            criterionType);

        auto transition = afToVector<float>(engine->criterion->param(0).array());
        decoder.reset(new WordLMDecoder(
            decoderOpt,
            trie,
            lm,
            silIdx,
            blankIdx,
            wordDict.getIndex(kUnkToken),
            transition));
    }
    ~WrapDecoder() {}

    char *decode(Emission *emission) {
        auto rawEmission = emission->emission;
        auto emissionVec = afToVector<float>(rawEmission);
        int N = rawEmission.dims(0);
        int T = rawEmission.dims(1);

        std::vector<float> score;
        std::vector<std::vector<int>> wordPredictions;
        std::vector<std::vector<int>> letterPredictions;
        auto results = decoder->decode(emissionVec.data(), T, N);
        auto wordPrediction = wrdIdx2Wrd(results[0].words, wordDict);
        auto words = join(" ", wordPrediction);
        return strdup(words.c_str());
    }

    std::shared_ptr<KenLM> lm;
    std::unique_ptr<Decoder> decoder;
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

bool w2l_engine_export(w2l_engine *engine, const char *path) {
    return reinterpret_cast<Engine *>(engine)->exportModel(path);
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
