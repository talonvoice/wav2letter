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

#include "w2l_encode.h"
#include "w2l_encode_backend.h"
#include "b2l.h"

using namespace w2l;

// helper functions
namespace {

void trim(std::string &s, std::string needle=" ") {
    s.erase(s.find_last_not_of(needle)+1);
    s.erase(s.begin(), s.begin() + s.find_first_not_of(needle));
}

std::pair<std::string, std::string> splitOn(std::string s, std::string on) {
    auto split = s.find(on);
    if (split == std::string::npos) {
        return {s, ""};
    }
    auto first = s.substr(0, split);
    auto second = s.substr(split + on.size());
    trim(first, ", ");
    trim(second, ", ");
    return {first, second};
}

std::vector<std::string> splitAll(std::string s, std::string on) {
    std::vector<std::string> out;
    if (on.empty()) {
        out.push_back(s);
        return out;
    }
    size_t pos = 0;
    while (1) {
        auto split = s.find(on, pos);
        auto first = s.substr(pos, split - pos);
        trim(first, ", ");
        out.push_back(first);
        if (split == std::string::npos) {
            break;
        }
        pos = split + on.size();
    }
    return out;
}

std::string findParens(std::string s, std::string left="(", std::string right=")") {
    auto start = s.find(left);
    auto end = s.find(right, start);
    auto sp = s.substr(start + 1, end - start - 1);
    trim(sp);
    return sp;
}

}

w2l_emission *afToEmission(af::array af) {
    int N = af.dims(0);
    int T = af.dims(1);
    size_t size = sizeof(w2l_emission) + sizeof(float) * N * T;
    w2l_emission *emission = (w2l_emission *)malloc(size);
    af.host(emission->matrix);
    return emission;
}

Engine::Engine() {
    loaded = false;
}

w2l_emission *Engine::forward(float *samples, size_t sample_count) {
    struct W2lLoaderData data = {};
    std::copy(samples, samples + sample_count, std::back_inserter(data.input));

    auto feat = featurize({data}, {});
    auto input = af::array(feat.inputDims, feat.input.data());
    auto rawEmission = network->forward({fl::input(input)}).front();
    return afToEmission(rawEmission.array());
}

af::array Engine::process(const af::array &features) {
    return network->forward({fl::input(features)}).front().array();
}

bool Engine::loadW2lModel(std::string acousticModelPath, std::string tokensPath) {
    // TODO: put feature type and params in model config section
    // FLAGS_mfsc = true;
    // gflags::FlagSaver flagsave; // TODO: don't clobber global flags (wordpiece seems to need global flags for now)
    W2lSerializer::load(acousticModelPath, config, network, criterion);
    auto flags = config.find(kGflags);
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    criterionType = FLAGS_criterion;
    network->eval();
    criterion->eval();
    tokenDict = Dictionary(tokensPath);
    if (criterionType == kCtcCriterion) {
        tokenDict.addEntry(kBlankToken);
    }
    loaded = true;
    return true;
}

bool Engine::loadB2lModel(std::string path) {
    auto file = b2l::File::open_file(path);

    // load main sections
    auto config = file.section("config").keyval();
    auto arch = w2l::split("\n", file.section("arch").utf8());
    criterionType = config["criterion"];

    auto tokenStream = std::istringstream(file.section("tokens").utf8());
    tokenDict = Dictionary(tokenStream);
    // TODO: ensure that resulting tokenDict.indexSize() > 0?
    if (criterionType == kCtcCriterion &&
            tokenDict.indexSize() > 0 &&
            tokenDict.getEntry(tokenDict.indexSize() - 1) != kBlankToken) {
        tokenDict.addEntry(kBlankToken);
    }

    auto flags = file.section("flags").keyval();
    std::ostringstream flagsfile;
    for (auto &pair : flags) {
        auto &key = pair.first;
        if (!key.empty() && !(key.size() == 1 && key[0] == '\0')) {
            flagsfile << "--" << pair.first << "=" << pair.second << "\n";
        }
    }
    // TODO: don't clobber global flags (wordpiece needs global flags for featurization)
    gflags::ReadFlagsFromString(flagsfile.str(), gflags::GetArgv0(), true);
    this->config[kGflags] = flagsfile.str();

    // create a blank model from the arch
    network = createW2lSeqModule(arch, getSpeechFeatureSize(), tokenDict.indexSize());
    // load the parameters
    auto layers = file.section("layers").layers();
    auto seq = dynamic_cast<fl::Sequential *>(network.get());
    auto modules = seq->modules();
    for (ssize_t i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];
        auto module = modules[i].get();
        /*
        std::cout << "loading layer: " << layer.arch << "\n";
        std::cout << "matching module: " << layerArch(module) << "\n";
        std::cout << "layer params=" << layer.params.size() << "\n";
        std::cout << "module params=" << module->params().size() << "\n";
        */
        for (ssize_t j = 0; j < layer.params.size(); j++) {
            auto &array = layer.params[j];
            switch (array.type()) {
                case b2l::Array::Type::FP32: {
                    auto val = array.array<float>();
                    fl::Variable v(af::array(module->param(j).array().dims(), val.data()), false);
                    module->setParams(v, j);
                    break;
                }
                default:
                    throw std::runtime_error("loading unsupported b2l parameter type");
            }
        }
    }
    // create blank criterion
    // TODO: stuff a serialized criterion in the b2l file?
    auto scalemode = CriterionScaleMode::NONE;
    if (criterionType == kCtcCriterion) {
        criterion = std::make_shared<CTCLoss>(scalemode);
    } else if (criterionType == kAsgCriterion) {
        criterion = std::make_shared<ASGLoss>(tokenDict.indexSize(), scalemode);
        // load transitions
        auto transitions = file.section("transitions").array<float>();
        fl::Variable v(af::array(tokenDict.indexSize(), tokenDict.indexSize(), transitions.data()), false);
        // need to use setParams to trigger ASGLoss->syncTransitions()
        criterion->setParams(v, 0);
    } else {
        throw std::runtime_error("unsupported criterion");
    }
    /* {
    } else if (criterionType == kSeq2SeqCriterion) {
      criterion = std::make_shared<Seq2SeqCriterion>(buildSeq2Seq(numClasses, tokenDict.getIndex(kEosToken)));
    } else if (criterionType == kTransformerCriterion) {
      criterion =
          std::make_shared<TransformerCriterion>(buildTransformerCriterion(
              numClasses,
              FLAGS_am_decoder_tr_layers,
              FLAGS_am_decoder_tr_dropout,
              FLAGS_am_decoder_tr_layerdrop,
              tokenDict.getIndex(kEosToken)));
    }
    */

    network->eval();
    criterion->eval();
    loaded = true;
    return true;
}

bool Engine::exportW2lModel(std::string path) {
    W2lSerializer::save(path, config, network, criterion);
    return true;
}

// TODO: catch C++ exceptions here, or in the C API wrapper
bool Engine::exportB2lModel(std::string path) {
    std::ostringstream arch;
    auto seq = dynamic_cast<fl::Sequential *>(network.get());
    auto modules = seq->modules();
    std::vector<b2l::Layer> layers;
    layers.reserve(modules.size());

    bool badLayer = false;
    for (auto &module : modules) {
        std::string layerString = layerArch(module.get());
        std::cerr << "[w2lapi] exporting: " << layerString << std::endl;
        if (layerString == "") {
            badLayer = true;
            continue;
        }
        arch << layerString << "\n";

        auto flParams = module->params();
        std::vector<b2l::Array> params;
        params.reserve(flParams.size());
        for (auto &var : flParams) {
            params.emplace_back(b2l::Array(afToVector<float>(var.array())));
        }
        layers.emplace_back(b2l::Layer{layerString, params});
    }
    if (badLayer) {
        std::cerr << "[w2lapi] aborting export due to layer errors" << std::endl;
        return false;
    }

    b2l::File file;
    // empty all path-related flags
    std::unordered_set<std::string> skip_flags{
        "train", "valid", "test", "archdir", "datadir", "rundir", "emission_dir", "log_dir",
        "tokens", "lexicon", "lm_vocab", "lm", "sclite", "rndv_filepath",
    };
    std::map<std::string, std::string> flags;
    std::string key, val;
    for (auto line : splitAll(config.find(kGflags)->second, "\n")) {
        std::tie(key, val) = splitOn(line, "=");
        trim(key, "-");
        if (skip_flags.find(key) != skip_flags.end()) {
            val = "";
        }
        flags[key] = val;
    }
    file.add_section("flags").keyval(flags);

    // TODO: feature params? mfsc / feature width / channels / samplerate / etc
    file.add_section("config").keyval({
        {"criterion", criterionType},
    });
    file.add_section("arch").utf8(arch.str());
    file.add_section("tokens").utf8(exportTokens());
    file.add_section("layers").layers(layers);

    if (criterionType == kAsgCriterion) {
        auto params = criterion->param(0);
        auto array = afToVector<float>(params.array());
        file.add_section("transitions").array(array);
    }

    auto writer = b2l::Writer::open_file(path);
    file.write_to(writer);
    return true;
}

std::vector<float> Engine::transitions() const {
    if (criterionType == kAsgCriterion) {
        return afToVector<float>(criterion->param(0).array());
    }
    return {};
}

// export functions
std::string Engine::exportTokens() {
    std::ostringstream ostr;
    for (int i = 0; i < tokenDict.indexSize(); i++) {
        ostr << tokenDict.getEntry(i);
        ostr << "\n";
    }
    return ostr.str();
}

// NOTE: The order and structure of these layer exporters should match W2lModule.cpp
std::string Engine::layerArch(fl::Module *module) {
    std::ostringstream ostr;
    auto pretty = module->prettyString();
    auto type = pretty.substr(0, pretty.find(" ("));

    // TYPE: TRANSFORMATIONS
    if (type == "Reorder") {
        auto dims = findParens(pretty);
        std::replace(dims.begin(), dims.end(), ',', ' ');
        ostr << "RO " << dims;
    } else if (type == "View") {
        auto ratio = findParens(pretty);
        ostr << "V " << findParens(pretty);
    } else if (type == "Padding") {
        auto params = findParens(pretty);
        std::string value;
        std::tie(value, params) = splitOn(params, ", ");

        // Padding (0, { (10, 0), })
        std::string l, r;
        ostr << "PD " << value;
        for (auto &seg : splitAll(params, "),")) {
            trim(seg, "{()} ");
            std::tie(l, r) = splitOn(seg, ", ");
            ostr << " " << l << " " << r;
        }

    // TYPE: TRANSFORMERS (TODO)
    // } else if (type == "Transformer") {
    // } else if (type == "PositionEmbedding") {

    // TYPE: CONVOLUTIONS
    } else if (type == "Conv2D") {
        // Conv2D (234->514, 23x1, 1,1, 0,0, 1, 1) (with bias)
        auto parens = findParens(pretty);
        bool bias = pretty.find("with bias") >= 0;
        std::string inputs, outputs, szX, szY, padX, padY, strideX, strideY, dilateX, dilateY;
        // TODO: I could get some of these from the params' dims instead of string parsing

        auto comma1 = parens.find(',') + 1;
        std::tie(inputs, outputs) = splitOn(parens.substr(0, comma1), "->");

        auto comma2 = parens.find(',', comma1) + 1;
        std::tie(szX, szY) = splitOn(parens.substr(comma1, comma2 - comma1 - 1), "x");

        auto comma4 = parens.find(',', parens.find(',', comma2) + 1) + 1;
        std::tie(strideX, strideY) = splitOn(parens.substr(comma2, comma4 - comma2 - 1), ",");

        auto comma6 = parens.find(',', parens.find(',', comma4) + 1) + 1;
        std::tie(padX, padY) = splitOn(parens.substr(comma4, comma6 - comma4 - 1), ",");
        if (padX == "SAME") padX = "-1";
        if (padY == "SAME") padY = "-1";

        auto comma8 = parens.find(',', parens.find(',', comma6) + 1) + 1;
        std::tie(dilateX, dilateY) = splitOn(parens.substr(comma6, comma8 - comma6 - 1), ",");

        // fl::Conv1D? C  [inputChannels] [outputChannels] [xFilterSz] [xStride] [xPadding <OPTIONAL>] [xDilation <OPTIONAL>]
        // fl::Conv2D  C2 [inputChannels] [outputChannels] [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding <OPTIONAL>] [yPadding <OPTIONAL>] [xDilation <OPTIONAL>] [yDilation <OPTIONAL>]
        if (szY == "1") {
            ostr << "C "  << inputs << " " << outputs << " " << szX << " " << strideX << " " << padX << " " << dilateX;
        } else {
            ostr << "C2 " << inputs  << " " << outputs
                << " " << szX     << " " << szY
                << " " << strideX << " " << strideY
                << " " << padX    << " " << padY
                << " " << dilateX << " " << dilateY;
        }
    // Time-Depth Separable Block (9, 60, 19) [1140 -> 1140 -> 1140]
    } else if (type == "Time-Depth Separable Block") {
        auto tds = static_cast<w2l::TDSBlock *>(module);
        auto tdsmods = tds->modules();
        auto seq1 = static_cast<fl::Sequential *>(tdsmods[0].get());

        // find dropout
        auto dropstr = seq1->modules().back()->prettyString();
        auto dropout = findParens(dropstr);

        // find right-pad
        auto padstr = seq1->modules()[0]->prettyString();
        auto params = findParens(padstr);
        auto rightPad = splitAll(params, ",")[2];

        // find lNormIncludeTime
        auto layernormstr = tdsmods[1]->prettyString();
        layernormstr = findParens(layernormstr, "{", "}");
        trim(layernormstr);
        bool lNormIncludeTime = (layernormstr == "0 1 2");

        // extract other params
        auto convW = tds->param(0);
        int kernelSize = convW.dims(0);
        int channels = convW.dims(2);
        auto linW = tds->param(4);
        int width = linW.dims(0) / channels;
        int linOuter = linW.dims(1);
        int linInner = linW.dims(0);
        if (linInner == linOuter) {
            linInner = 0;
        }
        ostr << "TDS " << channels << " " << kernelSize << " " << width;
        ostr << " " << dropout << " " << linInner << " " << rightPad;
        ostr << " " << (lNormIncludeTime ? "1" : "0");
    // } else if (type == "AsymmetricConv1D") {

    // TYPE: LINEAR
    } else if (type == "Linear") {
        std::string inputs, outputs;
        std::tie(inputs, outputs) = splitOn(findParens(pretty), "->");
        ostr << "L " << inputs << " " << outputs;

    // TYPE: EMBEDDING (TODO)

    // TYPE: NORMALIZATIONS
    // } else if (type == "BatchNorm") {
    } else if (type == "LayerNorm") {
        ostr << "LN " << findParens(pretty, "{", "}");
    } else if (type == "WeightNorm") {
        auto wn = dynamic_cast<fl::WeightNorm *>(module);
        auto lastParam = pretty.rfind(",") + 2;
        auto dim = pretty.substr(lastParam, pretty.size() - lastParam - 1);
        ostr << "WN " << dim << " ";
        ostr << layerArch(wn->module().get());
    } else if (type == "Dropout") {
        ostr << "DO " << findParens(pretty);

    // TYPE: POOLING (TODO)

    // TYPE: ACTIVATIONS
    } else if (type == "ELU") {
        ostr << "ELU";
    } else if (type == "ReLU") {
        ostr << "R";
    } else if (type == "ReLU6") {
        ostr << "R6";
    // } else if (type == "PReLU") {
    } else if (type == "Log") {
        ostr << "LG";
    } else if (type == "HardTanh") {
        ostr << "HT";
    } else if (type == "Tanh") {
        ostr << "T";
    } else if (type == "GatedLinearUnit") {
        ostr << "GLU " << findParens(pretty);
    // } else if (type == "LogSoftmax") {
    // } else if (type == "Swish") {

    // TYPE: RNNs (TODO)
    // TYPE: Residual block (TODO)

    // TYPE: Data Augmentation
    } else if (type == "SpecAugment") {
        // SpecAugment ( W: 60, F: 18, mF: 2, T: 100, p: 0.05, mT: 2 )
        auto cs = splitAll(pretty, ", ");
        for (auto &s : cs) {
            s = splitOn(s, ": ").second;
            trim(s, " )");
        }
        ostr << "SAUG " << cs[0] << " " << cs[1] << " " << cs[2] << " " << cs[3] << " " << cs[4] << " " << cs[5];

    // TYPE: Unknown
    } else {
        std::cerr << "[w2lapi] error, unknown layer type: " << type << std::endl;
        return "";
    }
    return ostr.str();
}
