#include <flashlight/autograd/Variable.h>
#include "libraries/common/Dictionary.h"

namespace w2l {
class SequenceCriterion;
}

namespace fl {
class Module;
class Variable;
}

using namespace w2l;

class Engine {
public:
    Engine();
    ~Engine() {}

    w2l_emission *forward(float *samples, size_t sample_count);
    af::array process(const af::array &features);

    bool loadW2lModel(std::string modelPath, std::string tokensPath);
    bool loadB2lModel(std::string path);
    bool exportW2lModel(std::string path);
    bool exportB2lModel(std::string path);

    std::vector<float> transitions() const;
    af::array viterbiPath(const af::array &data) const;

private:
    std::string exportTokens();
    std::string layerArch(fl::Module *module);

private:
    bool loaded;
    std::unordered_map<std::string, std::string> config;
    std::shared_ptr<fl::Module> network;
    std::shared_ptr<SequenceCriterion> criterion;
    std::string criterionType;
    Dictionary tokenDict;
};
