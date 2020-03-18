cmake_minimum_required(VERSION 3.5.1)
project(libw2l)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_directories(${PROJECT_SOURCE_DIR}/../opt/lib)
include_directories(${PROJECT_SOURCE_DIR}/../opt/include)
include_directories(inference/)
add_definitions(-flto=thin)

find_library(LIBOMP NAMES libomp.a)
find_package(GFLAGS REQUIRED)

set(COMMON_SRC
    src/common/Defines.cpp
    src/common/Dictionary.cpp
    src/common/Transforms.cpp
    src/common/Utils.cpp
    src/common/Utils-base.cpp
)

set(FEATURE_SRC
    src/libraries/feature/Ceplifter.cpp
    src/libraries/feature/Dct.cpp
    src/libraries/feature/Derivatives.cpp
    src/libraries/feature/Dither.cpp
    src/libraries/feature/Mfcc.cpp
    src/libraries/feature/Mfsc.cpp
    src/libraries/feature/PowerSpectrum.cpp
    src/libraries/feature/PreEmphasis.cpp
    src/libraries/feature/SpeechUtils.cpp
    src/libraries/feature/TriFilterbank.cpp
    src/libraries/feature/Windowing.cpp
)

set(INFERENCE_SRC
    inference/inference/module/InferenceModule.cpp
    inference/inference/module/nn/Linear.cpp
    inference/inference/module/nn/Residual.cpp
    inference/inference/module/nn/LayerNorm.cpp
    inference/inference/module/nn/TDSBlock.cpp
    inference/inference/module/nn/Relu.cpp
    inference/inference/module/nn/Sequential.cpp
    inference/inference/module/nn/Identity.cpp
    inference/inference/module/nn/backend/fbgemm/Conv1dFbGemm.cpp
    inference/inference/module/nn/backend/fbgemm/LinearFbGemm.cpp
    inference/inference/module/nn/backend/fbgemm/PackedGemmMatrixFP16.cpp
    inference/inference/module/nn/Conv1d.cpp
    inference/inference/module/nn/LocalNorm.cpp
    inference/inference/module/ModuleProcessingState.cpp
    inference/inference/module/ModuleParameter.cpp
    inference/inference/module/feature/LogMelFeature.cpp
    inference/inference/common/IOBuffer.cpp
    inference/inference/common/DataType.cpp
    inference/inference/common/DefaultMemoryManager.cpp
    inference/inference/common/Functions.cpp
)

find_library(LIBIPPS NAMES libipps.a HINTS /opt/intel/ipp/lib)
find_library(LIBIPPVM NAMES libippvm.a HINTS /opt/intel/ipp/lib)
find_library(LIBIPPCORE NAMES libippcore.a HINTS /opt/intel/ipp/lib)
set(IPP_LIBS ${LIBIPPS} ${LIBIPPCORE} ${LIBIPPVM})

add_library(w2lstream SHARED
    ${FEATURE_SRC}
    ${INFERENCE_SRC}
    w2lstream.cpp
)
target_link_libraries(w2lstream
    ${IPP_LIBS}
    ${LIBOMP}
    ${GFLAGS_LIBRARIES}
    ${LIBOMP}
    lzma
    bz2
    z

    # fbgemm
    fbgemm
    cpuinfo
    asmjit
    clog
    pthread
)
if (APPLE)
    target_link_libraries(w2lstream -Wl,-cache_path_lto,/tmp/lto-cache)
    add_definitions(-Xpreprocessor -fopenmp)
endif()

target_include_directories(
  w2lstream
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${fbgemm_INCLUDE_DIRS}
)

add_executable(wavstream wavstream.c)
target_link_libraries(wavstream w2lstream)

# add_executable(simple_example
#     inference/inference/examples/SimpleStreamingASRExample.cpp
#     inference/inference/examples/AudioToWords.cpp
#     inference/inference/examples/Util.cpp
# )
# target_link_libraries(simple_example w2lstream)

install(TARGETS w2lstream DESTINATION lib)
install(FILES w2lstream.h DESTINATION include)
