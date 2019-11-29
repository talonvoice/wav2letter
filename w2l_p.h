#include <flashlight/autograd/Variable.h>

namespace w2l {
class SequenceCriterion;
}

namespace fl {
class Module;
class Variable;
}

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
    Emission(EngineBase *engine, af::array emission, af::array inputs);
    ~Emission() {}

    char *text();

    EngineBase *engine;
    af::array emission;
    af::array inputs;
};

class Engine : public EngineBase {
public:
    Engine(const char *acousticModelPath, const char *tokensPath);
    ~Engine() {}

    Emission *process(float *samples, size_t sample_count);
    af::array process(const af::array &features);

    bool exportModel(const char *path);

    std::vector<float> transitions() const;

    af::array viterbiPath(const af::array &data) const;

private:
    std::tuple<std::string, std::string> splitOn(std::string s, std::string on);
    std::string findParens(std::string s);
    void exportParams(std::ofstream& f, fl::Variable params);
    bool exportLayer(std::ofstream& f, fl::Module *module);
    void exportTransitions(std::ofstream& f);
    void exportTokens(std::ofstream& f);
};
