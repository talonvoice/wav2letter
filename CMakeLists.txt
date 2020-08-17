cmake_minimum_required(VERSION 3.5.1)
project(libw2l)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# link_directories(${PROJECT_SOURCE_DIR}/../opt/lib)
include_directories(${PROJECT_SOURCE_DIR}/../opt/include)
include_directories(inference/ src/)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    find_library(GFLAGS_LIBRARIES NAMES libgflags_static.a)
    set(GFLAGS_LIBRARIES ${GFLAGS_LIBRARIES} shlwapi)
    set(LIBOMP "")
else()
    add_definitions(-flto=thin)
    find_package(GFLAGS REQUIRED)
    find_library(LIBOMP NAMES libomp.a)
endif()

find_package(kenlm REQUIRED)
set(KENLM_MAX_ORDER 6 CACHE STRING "KENLM_MAX_ORDER")
add_definitions(-DBMR_USE_QUICKSELECT -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})

if (NOT SKIP_W2L)
    # libw2l-only libraries:
    find_package(GLOG REQUIRED)
    find_package(Threads REQUIRED)
    find_package(ArrayFire REQUIRED)
    find_package(flashlight REQUIRED)
    if (flashlight_FOUND)
        message(STATUS "flashlight found (include: ${FLASHLIGHT_INCLUDE_DIRS} lib: flashlight::flashlight )")
    else()
        message(FATAL_ERROR "flashlight not found")
    endif()
    # end libw2l-only
endif()

set(COMMON_SRC
    src/common/Defines.cpp
    src/common/Transforms.cpp
    src/common/Utils.cpp

    src/libraries/common/Dictionary.cpp
    src/libraries/common/Utils.cpp
    src/libraries/common/WordUtils.cpp
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

set(DECODER_SRC
    src/libraries/decoder/LexiconDecoder.cpp
    src/libraries/decoder/LexiconFreeDecoder.cpp
    src/libraries/decoder/LexiconFreeSeq2SeqDecoder.cpp
    src/libraries/decoder/LexiconSeq2SeqDecoder.cpp
    src/libraries/decoder/Trie.cpp
    src/libraries/decoder/Utils.cpp

    src/libraries/lm/ConvLM.cpp
    src/libraries/lm/KenLM.cpp
    src/libraries/lm/ZeroLM.cpp
)

set(DATA_SRC
    src/data/Featurize.cpp
    # src/data/FeatureTransforms.cpp
    src/data/Utils.cpp

    # src/data/ListFileDataset.cpp
    # src/data/Sound.cpp
    # src/data/W2lBlobsDataset.cpp
    # src/data/W2lDataset.cpp
    # src/data/W2lListFilesDataset.cpp
)

set(INFERENCE_SRC
    inference/inference/common/DataType.cpp
    inference/inference/common/DefaultMemoryManager.cpp
    inference/inference/common/Functions.cpp
    inference/inference/common/IOBuffer.cpp
    inference/inference/decoder/Decoder.cpp
    inference/inference/module/InferenceModule.cpp
    inference/inference/module/ModuleParameter.cpp
    inference/inference/module/ModuleProcessingState.cpp
    inference/inference/module/feature/LogMelFeature.cpp
    inference/inference/module/nn/Conv1d.cpp
    inference/inference/module/nn/Identity.cpp
    inference/inference/module/nn/LayerNorm.cpp
    inference/inference/module/nn/Linear.cpp
    inference/inference/module/nn/LocalNorm.cpp
    inference/inference/module/nn/Relu.cpp
    inference/inference/module/nn/Residual.cpp
    inference/inference/module/nn/Sequential.cpp
    inference/inference/module/nn/TDSBlock.cpp
    inference/inference/module/nn/backend/fbgemm/Conv1dFbGemm.cpp
    inference/inference/module/nn/backend/fbgemm/LinearFbGemm.cpp
    inference/inference/module/nn/backend/fbgemm/PackedGemmMatrixFP16.cpp
)

set(MODULE_SRC
    src/module/ConvLmModule.cpp
    src/module/SpecAugment.cpp
    src/module/TDSBlock.cpp
    src/module/W2lModule.cpp
)

set(VITERBI_SRC
    src/libraries/criterion/cpu/ViterbiPath.cpp
)

set(CRITERION_SRC
    src/criterion/ConnectionistTemporalClassificationCriterion.cpp
    src/criterion/CriterionUtils.cpp
    src/criterion/ForceAlignmentCriterion.cpp
    src/criterion/FullConnectionCriterion.cpp
    src/criterion/Seq2SeqCriterion.cpp
    src/criterion/TransformerCriterion.cpp
    src/criterion/attention/ContentAttention.cpp
    src/criterion/attention/LocationAttention.cpp
    src/criterion/attention/MedianWindow.cpp
    src/criterion/attention/MultiHeadAttention.cpp
    src/criterion/attention/SoftPretrainWindow.cpp
    src/criterion/attention/SoftWindow.cpp
    src/criterion/attention/StepWindow.cpp
    src/criterion/backend/cpu/ConnectionistTemporalClassificationCriterion.cpp
    src/criterion/backend/cpu/CriterionUtils.cpp
    src/criterion/backend/cpu/ForceAlignmentCriterion.cpp
    src/criterion/backend/cpu/FullConnectionCriterion.cpp

    src/libraries/criterion/cpu/ConnectionistTemporalClassificationCriterion.cpp
    src/libraries/criterion/cpu/CriterionUtils.cpp
    src/libraries/criterion/cpu/FullConnectionCriterion.cpp
    src/libraries/criterion/cpu/ForceAlignmentCriterion.cpp
    src/libraries/criterion/cpu/ViterbiPath.cpp
)

find_library(LIBIPPS NAMES libipps.a HINTS /opt/intel/ipp/lib)
find_library(LIBIPPVM NAMES libippvm.a HINTS /opt/intel/ipp/lib)
find_library(LIBIPPCORE NAMES libippcore.a HINTS /opt/intel/ipp/lib)
set(IPP_LIBS ${LIBIPPS} ${LIBIPPCORE} ${LIBIPPVM})

# w2l target
if (NOT SKIP_W2L)
    add_library(w2l SHARED
        ${COMMON_SRC}
        ${FEATURE_SRC}
        ${DECODER_SRC}
        ${MODULE_SRC}
        ${CRITERION_SRC}
        ${DATA_SRC}

        w2l/w2l_encode.cpp
        w2l/w2l_decode.cpp
        w2l/w2l_encode_backend.cpp
    )
    if (USE_MKL)
        target_compile_definitions(w2l PUBLIC -DUSE_MKL -DW2L_LIBRARIES_USE_MKL)
        find_library(MKL_CORE NAMES libmkl_core.a)
        find_library(MKL_INTEL_LP64 NAMES libmkl_intel_lp64.a)
        find_library(MKL_GNU_THREAD NAMES libmkl_gnu_thread.a)
        target_link_libraries(w2l "-Wl,--start-group" ${MKL_CORE} ${MKL_INTEL_LP64} ${MKL_GNU_THREAD} "-Wl,--end-group")
    else()
        target_link_libraries(w2l "-framework Accelerate")
    endif()

    set_target_properties(ArrayFire::afcpu PROPERTIES INTERFACE_LINK_LIBRARIES "")
    target_link_libraries(w2l
        "-L${PROJECT_SOURCE_DIR}/../opt/lib"
        ${KENLM_LIBRARIES}
        ${IPP_LIBS}
        flashlight::flashlight
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${LIBOMP}
    )
    if (APPLE)
        target_link_libraries(w2l -Wl,-cache_path_lto,/tmp/lto-cache)
        target_compile_options(w2l -Xpreprocessor -fopenmp)
    endif()

    target_include_directories(
      w2l
      PUBLIC
      ${PROJECT_SOURCE_DIR}/src
      ${KENLM_INCLUDE_DIRS}
      ${FLASHLIGHT_INCLUDE_DIRS}
    )
endif() # SKIP_W2L

# w2ldecode target
if (NOT SKIP_W2LDECODE)
    add_library(w2ldecode SHARED
        ${COMMON_SRC}
        ${DECODER_SRC}
        ${VITERBI_SRC}
        w2l/w2l_decode.cpp
    )

    target_link_libraries(w2ldecode
        "-L${PROJECT_SOURCE_DIR}/../opt/lib"
        ${KENLM_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${LIBOMP}
    )
    if (APPLE)
        target_link_libraries(w2ldecode -Wl,-cache_path_lto,/tmp/lto-cache)
        target_compile_options(w2ldecode -Xpreprocessor -fopenmp)
    elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
        # no openmp on windows yet
    else()
        target_compile_options(w2ldecode -fopenmp)
    endif()
    target_include_directories(
      w2ldecode
      PUBLIC
      ${PROJECT_SOURCE_DIR}/src
      ${KENLM_INCLUDE_DIRS}
      ${FLASHLIGHT_INCLUDE_DIRS}
    )
endif() # SKIP_W2LDECODE

# w2lstream target
if (NOT SKIP_W2LSTREAM)
    add_library(w2lstream SHARED
        ${COMMON_SRC}
        ${FEATURE_SRC}
        ${INFERENCE_SRC}
        ${DECODER_SRC}
        w2l/w2lstream.cpp
    )
    target_link_libraries(w2lstream
        "-L${PROJECT_SOURCE_DIR}/../opt/lib"
        ${GFLAGS_LIBRARIES}
        ${IPP_LIBS}
        ${KENLM_LIBRARIES}
        ${LIBOMP}

        # fbgemm
        fbgemm
        cpuinfo
        asmjit
        clog
        pthread
        dl
    )
    if (APPLE)
        target_link_libraries(w2lstream -Wl,-cache_path_lto,/tmp/lto-cache)
        add_definitions(-Xpreprocessor -fopenmp)
        target_link_libraries(w2lstream "-framework Accelerate")
    elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
        # no openmp on windows yet
    else()
        target_compile_options(w2ldecode -fopenmp)
    endif()

    target_include_directories(
      w2lstream
      PUBLIC
      ${PROJECT_SOURCE_DIR}/src
      ${KENLM_INCLUDE_DIRS}
    )

    add_executable(wavstream w2l/wavstream.c)
    target_link_libraries(wavstream w2lstream)
endif() # SKIP_W2LSTREAM

if (NOT SKIP_W2L)
    add_executable(w2l_cli w2l/cli.cpp)
    target_link_libraries(w2l_cli w2l dl)
endif()

# add_executable(simple_example
#     inference/inference/examples/SimpleStreamingASRExample.cpp
#     inference/inference/examples/AudioToWords.cpp
#     inference/inference/examples/Util.cpp
# )
# target_link_libraries(simple_example w2lstream)

install(TARGETS w2ldecode DESTINATION lib)
install(FILES   w2l/w2l_common.h w2l/w2l_decode.h DESTINATION include)
