// Verify OpenVDB and tiny-cuda-nn linkage by exercising basic APIs.
#include <iostream>

#if DVREN_HAS_OPENVDB
#  include <openvdb/openvdb.h>
#endif

#if DVREN_WITH_TCNN
#  include <tiny-cuda-nn/common.h>
#endif

int main() {
#if DVREN_HAS_OPENVDB
    // Initialize OpenVDB and create a simple grid
    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(/*background*/ 0.0f);
    grid->setName("TestGrid");
    std::cout << "OpenVDB initialized. Grid name: " << grid->getName() << "\n";
#else
    std::cout << "OpenVDB disabled at build time.\n";
#endif

#if DVREN_WITH_TCNN
    // Touch tiny-cuda-nn symbols via headers. Print a constexpr from the library.
    std::cout << "tiny-cuda-nn available. MIN_GPU_ARCH=" << tcnn::MIN_GPU_ARCH << "\n";
#else
    std::cout << "tiny-cuda-nn disabled at build time.\n";
#endif

    std::cout << "dvren startup complete." << std::endl;
    return 0;
}
