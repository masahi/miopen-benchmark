#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"

void VGGNet() {
    TensorDesc input_dim(16, 3, 224, 224);

    Sequential features(input_dim);
    /* features */

    std::vector<std::vector<int>> arches{{2,64}, {2,128}, {4,256}, {4, 512}, {4, 512}};

    for(auto& arch: arches){
      int num_conv = arch[0];
      int num_output = arch[1];
      for(int i = 0; i < num_conv; ++i) {
        features.addConv(num_output, 3, 1, 1);
        features.addReLU();
      }
      features.addMaxPool(2, 0, 2);
    }

    DEBUG("Dims after Features: " << features.getOutputDesc());

    /* classifier */
    Sequential classifier(features.getOutputDesc());
    auto desc = features.getOutputDesc();
    // TODO Dropout
    classifier.reshape(input_dim.n, desc.c * desc.h * desc.w, 1, 1);
    classifier.addLinear(4096);
    classifier.addReLU();
    //add dropout
    // TODO: Dropout
    classifier.addLinear(4096);
    classifier.addReLU();
    //add dropout
    classifier.addLinear(1000);
    classifier.addSoftmax();
    //add softmax

    Model m(input_dim);
    m.add(features);
    m.add(classifier);

    BenchmarkLogger::new_session("alex_net");
    BenchmarkLogger::benchmark(m, 50);
}

int main(int argc, char *argv[])
{
    device_init();

    // enable profiling
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    VGGNet();

    miopenDestroy(mio::handle());
    return 0;
}
