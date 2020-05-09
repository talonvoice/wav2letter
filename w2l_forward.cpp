#include <iostream>
#include <stdlib.h>
#include <string>
#include <typeinfo>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "data/W2lDataset.h"
#include "libraries/common/Dictionary.h"
#include "libraries/decoder/Decoder.h"
#include "libraries/decoder/Trie.h"
#include "libraries/decoder/Utils.h"
#include "libraries/lm/KenLM.h"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"

#include "w2l.h"
#include "w2l_p.h"

using namespace w2l;

Emission::Emission(EngineBase *engine, af::array emission, af::array inputs) {
    this->engine = engine;
    this->emission = emission;
    this->inputs = inputs;
}

char *Emission::text() {
    auto tokenPrediction =
        afToVector<int>(engine->criterion->viterbiPath(emission));
    auto letters = tknPrediction2Ltr(tokenPrediction, engine->tokenDict);
    if (letters.size() > 0) {
        std::ostringstream ss;
        for (auto s : letters) ss << s;
        return strdup(ss.str().c_str());
    }
    return strdup("");
}

Engine::Engine(const char *acousticModelPath, const char *tokensPath) {
    // TODO: set criterionType "correctly"
    W2lSerializer::load(acousticModelPath, config, network, criterion);
    auto flags = config.find(kGflags);
    // loading flags globally like this is gross, only way to work around it will be parameterizing everything about wav2letter better
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    criterionType = FLAGS_criterion;
    network->eval();
    criterion->eval();

    tokenDict = Dictionary(tokensPath);
    if (criterionType == kCtcCriterion) {
        tokenDict.addEntry(kBlankToken);
    }
    numClasses = tokenDict.indexSize();
}

Emission *Engine::process(float *samples, size_t sample_count) {
    struct W2lLoaderData data = {};
    std::copy(samples, samples + sample_count, std::back_inserter(data.input));

    auto feat = featurize({data}, {});
    auto input = af::array(feat.inputDims, feat.input.data());
    auto rawEmission = network->forward({fl::input(input)}).front();
    return new Emission(this, rawEmission.array(), input);
}

af::array Engine::process(const af::array &features) {
    return network->forward({fl::input(features)}).front().array();
}

bool Engine::exportModel(const char *path) {
    std::ofstream outfile;
    outfile.open(path, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
            std::cout << "[w2lapi] error, could not open file '" << path << "' (aborting export)" << std::endl;
        return false;
    }

    auto seq = dynamic_cast<fl::Sequential *>(network.get());
    exportTokens(outfile);
    exportTransitions(outfile);
    for (auto &module : seq->modules()) {
        if (!exportLayer(outfile, module.get())) {
            std::cout << "[w2lapi] aborting export" << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<float> Engine::transitions() const {
    if (criterionType == kAsgCriterion) {
        return afToVector<float>(criterion->param(0).array());
    }
    return {};
}

af::array Engine::viterbiPath(const af::array &data) const {
    return criterion->viterbiPath(data);
}

void trim(std::string &s, std::string needle) {
    s.erase(s.find_last_not_of(needle)+1);
    s.erase(s.begin(), s.begin() + s.find_first_not_of(needle));
}

std::tuple<std::string, std::string> Engine::splitOn(std::string s, std::string on) {
    auto split = s.find(on);
    auto first = s.substr(0, split);
    auto second = s.substr(split + on.size());
    trim(first, ", ");
    trim(second, ", ");
    return {first, second};
}

std::string Engine::findParens(std::string s) {
    auto start = s.find('(');
    auto end = s.find(')', start);
    auto sp = s.substr(start + 1, end - start - 1);
    return sp;
}

void Engine::exportParams(std::ofstream& f, fl::Variable params) {
    auto array = afToVector<float>(params.array());
    for (float& p : array) {
        f << std::hex << (uint32_t&)p;
        if (&p != &array.back()) {
            f << " ";
        }
    }
    f << std::dec;
}

void Engine::exportTokens(std::ofstream& f) {
    std::cout << "[w2lapi] exporting: tokens" << std::endl;
    f << "TOK ";
    for (int i = 0; i < tokenDict.indexSize(); i++) {
        if (i > 0) f << ";";
        std::string entry = tokenDict.getEntry(i);
        for (int i = 0; i < entry.size(); i++) {
            if (i > 0) f << ",";
            f << std::hex << (unsigned int)entry[i];
        }
    }
    f << std::dec;
    f << "\n";
}

void Engine::exportTransitions(std::ofstream& f) {
    std::cout << "[w2lapi] exporting: transitions" << std::endl;
    f << "ASG ";
    exportParams(f, criterion->param(0));
    f << "\n";
}

bool Engine::exportLayer(std::ofstream& f, fl::Module *module) {
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
        std::string inputs, outputs, szX, szY, padX, padY, strideX, strideY, dilateX, dilateY;
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

        // fl::Conv1D? C  [inputChannels] [outputChannels] [xFilterSz] [xStride] [xPadding <OPTIONAL>] [xDilation <OPTIONAL>]
        // fl::Conv2D  C2 [inputChannels] [outputChannels] [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding <OPTIONAL>] [yPadding <OPTIONAL>] [xDilation <OPTIONAL>] [yDilation <OPTIONAL>]
        if (szY == "1") {
            f << "C "  << inputs << " " << outputs << " " << szX << " " << strideX << " " << padX << " " << dilateX << " | ";
        } else {
            f << "C2 " << inputs  << " " << outputs
                << " " << szX     << " " << szY
                << " " << strideX << " " << strideY
                << " " << padX    << " " << padY
                << " " << dilateX << " " << dilateY << " | ";
        }
        exportParams(f, module->param(0));
        if (bias) {
            f << " | ";
            exportParams(f, module->param(1));
        }
        f << "\n";
    } else if (type == "Linear") {
        std::string inputs, outputs;
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

