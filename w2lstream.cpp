#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include "inference/decoder/Decoder.h"
#include "inference/examples/AudioToWords.h"
#include "inference/examples/Util.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"

extern "C" {
#include "w2lstream.h"
}

using namespace w2l;

class GreedyCTC {
public:
    GreedyCTC(std::vector<std::string> &tokens) : tokens(tokens) {
    }

    void run(float *data, size_t size) {
        if (size % tokens.size() != 0) {
            abort();
        }
        output.reserve(output.size() + size / tokens.size());
        for (int t = 0; t < size; t += tokens.size()) {
            int maxIdx = 0;
            float maxValue = data[t + 0];
            for (int n = 1; n < tokens.size(); n++) {
                float value = data[t + n];
                if (value > maxValue) {
                    maxIdx = n;
                    maxValue = value;
                }
            }
            // std::cout << "maxIdx=" << maxIdx << " maxValue=" << maxValue << std::endl;
            output.push_back(maxIdx);
        }
    }

    std::string text() {
        std::ostringstream ostr;
        int last = -1;
        for (int tok : output) {
            if (tok != tokens.size() - 1 && tok != last) {
                // ostr << " " << tokens[tok] << " (" << tok << ")";
                std::string s = tokens[tok];
                if (s[0] == '_') {
                    ostr << " " << s.substr(1);
                } else {
                    ostr << s;
                }
            }
            last = tok;
        }
        output.clear();
        return ostr.str();
    }

    std::vector<int> output;
    std::vector<std::string> &tokens;
};

class W2lStream {
public:
    W2lStream(const char *feature_model_path, const char *acoustic_model_path, const char *tokens_path, int chunk_size_arg) {
        chunk_size = chunk_size_arg;

        std::shared_ptr<streaming::Sequential> featureModule;
        std::shared_ptr<streaming::Sequential> acousticModule;

        {
            std::ifstream featureFile(feature_model_path);
            if (!featureFile.is_open()) {
                error = "cannot open feature file";
                return;
            }
            cereal::BinaryInputArchive ar(featureFile);
            ar(featureModule);
        }

        {
            std::ifstream amFile(acoustic_model_path);
            if (!amFile.is_open()) {
                error = "cannot open am file";
                return;
            }
            cereal::BinaryInputArchive ar(amFile);
            ar(acousticModule);
        }

        {
            std::ifstream tknFile(tokens_path);
            if (!tknFile.is_open()) {
                error = "cannot open tokens";
                return;
            }
            std::string line;
            while (std::getline(tknFile, line)) {
                tokens.push_back(line);
            }
        }

        dnnModule = std::make_shared<streaming::Sequential>();
        dnnModule->add(featureModule);
        dnnModule->add(acousticModule);
        decoder = new GreedyCTC(tokens);

        input = std::make_shared<streaming::ModuleProcessingState>(1);
        output = dnnModule->start(input);
    }

    ~W2lStream() {
        delete decoder;
    }

    std::string run(float *samples, size_t sample_count) {
        auto inputBuffer = input->buffer(0);
        auto outputBuffer = output->buffer(0);

        for (ssize_t i = 0; i < sample_count; i += chunk_size) {
            ssize_t size = std::min((ssize_t)chunk_size, (ssize_t)sample_count);
            inputBuffer->ensure<float>(size);
            float *data = inputBuffer->data<float>();
            std::copy(samples + i, samples + i + size, data);
            inputBuffer->move<float>(size);
            if (sample_count - i >= chunk_size) {
                dnnModule->run(input);
            } else {
                dnnModule->finish(input);
            }
            decoder->run(outputBuffer->data<float>(), outputBuffer->size<float>());
        }
        outputBuffer->consume<float>(outputBuffer->size<float>());
        return decoder->text();
    }

public:
    std::string error;

private:
    int chunk_size;
    std::shared_ptr<streaming::Sequential> dnnModule;
    std::shared_ptr<streaming::ModuleProcessingState> input, output;

    std::vector<std::string> tokens;
    GreedyCTC *decoder;
};

extern "C" {

w2lstream *w2lstream_new(const char *feature_model_path, const char *acoustic_model_path, const char *tokens_path, int chunk_size) {
    auto stream = new W2lStream(feature_model_path, acoustic_model_path, tokens_path, chunk_size);
    if (stream->error != "") {
        return NULL;
    }
    return reinterpret_cast<w2lstream *>(stream);
}

char *w2lstream_run(w2lstream *engine, float *samples, size_t sample_count) {
    auto stream = reinterpret_cast<W2lStream *>(engine);
    auto text = stream->run(samples, sample_count);
    return strdup(text.c_str());
}

void w2lstream_free(w2lstream *engine) {
    auto stream = reinterpret_cast<W2lStream *>(engine);
    delete stream;
}

} // extern "C"
