cmake_minimum_required(VERSION 3.5.1)
project(libw2l)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_directories(../opt/lib)
include_directories(/opt/intel/ipp/include)
add_definitions(-flto=thin)

find_library(LIBOMP NAMES libomp.a)
find_package(GLOG REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(Threads REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(flashlight REQUIRED)
if (flashlight_FOUND)
    message(STATUS "flashlight found (include: ${FLASHLIGHT_INCLUDE_DIRS} lib: flashlight::flashlight )")
else()
    message(FATAL_ERROR "flashlight not found")
endif()

set(COMMON_SRC
    src/common/Defines.cpp
    src/common/Dictionary.cpp
    src/common/Transforms.cpp
    src/common/Utils.cpp
    src/common/Utils-base.cpp
)

set(RUNTIME_SRC
    src/runtime/Logger.cpp
    src/runtime/Serial.cpp
    src/runtime/SpeechStatMeter.cpp
    src/runtime/Optimizer.cpp
)

set(MODULE_SRC
    src/module/SpecAugment.cpp
    src/module/TDSBlock.cpp
    src/module/W2lModule.cpp
)

set(DECODER_SRC
    src/decoder/Seq2SeqDecoder.cpp
    src/decoder/LexiconDecoder.cpp
    src/decoder/LexiconFreeDecoder.cpp
    src/decoder/TokenLMDecoder.cpp
    src/decoder/Trie.cpp
    src/decoder/Utils.cpp
    src/decoder/WordLMDecoder.cpp
)

set(CRITERION_SRC
    src/criterion/ConnectionistTemporalClassificationCriterion.cpp
    src/criterion/CriterionUtils.cpp
    src/criterion/ConnectionistTemporalClassificationCriterion.cpp
    src/criterion/CriterionUtils.cpp
    src/criterion/ForceAlignmentCriterion.cpp
    src/criterion/Seq2SeqCriterion.cpp
    src/criterion/FullConnectionCriterion.cpp

    src/criterion/attention/ContentAttention.cpp
    src/criterion/attention/LocationAttention.cpp
    src/criterion/attention/MedianWindow.cpp
    src/criterion/attention/SoftPretrainWindow.cpp
    src/criterion/attention/SoftWindow.cpp
    src/criterion/attention/StepWindow.cpp

    src/criterion/backend/cpu/ConnectionistTemporalClassificationCriterion.cpp
    src/criterion/backend/cpu/CriterionUtils.cpp
    src/criterion/backend/cpu/ForceAlignmentCriterion.cpp
    src/criterion/backend/cpu/FullConnectionCriterion.cpp
)

set(LIBRARIES_SRC
    src/libraries/criterion/cpu/CriterionUtils.cpp
    src/libraries/criterion/cpu/ForceAlignmentCriterion.cpp
    src/libraries/criterion/cpu/FullConnectionCriterion.cpp
    src/libraries/criterion/cpu/ViterbiPath.cpp

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

find_library(LIBIPPS NAMES libipps.a HINTS /opt/intel/ipp/lib)
find_library(LIBIPPVM NAMES libippvm.a HINTS /opt/intel/ipp/lib)
find_library(LIBIPPCORE NAMES libippcore.a HINTS /opt/intel/ipp/lib)
set(IPP_LIBS ${LIBIPPS} ${LIBIPPCORE} ${LIBIPPVM})
set(DATA_SRC
    src/data/Utils.cpp
    src/data/Featurize.cpp
)

find_package(kenlm REQUIRED)
set(KENLM_MAX_ORDER 6 CACHE STRING "KENLM_MAX_ORDER")
add_definitions(-DBMR_USE_QUICKSELECT -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})
set(LM_SRC
    src/lm/KenLM.cpp
    src/lm/ConvLM.cpp
)

add_library(w2l SHARED
    ${COMMON_SRC}
    ${RUNTIME_SRC}
    ${MODULE_SRC}
    ${DECODER_SRC}
    ${CRITERION_SRC}
    ${DATA_SRC}
    ${LIBRARIES_SRC}
    ${LM_SRC}
    w2l.cpp
    w2l_forward.cpp
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
    ${KENLM_LIBRARIES}
    ${IPP_LIBS}
    ${OPENMP_LIBRARIES}
    flashlight::flashlight
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${LIBOMP}
    lzma
    bz2
    z
)
if (APPLE)
    target_link_libraries(w2l -Wl,-cache_path_lto,/tmp/lto-cache)
endif()

target_include_directories(
  w2l
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${KENLM_INCLUDE_DIRS}
  ${FLASHLIGHT_INCLUDE_DIRS}
)

add_executable(w2lexport w2lexport.c)
target_link_libraries(w2lexport w2l)

install(TARGETS w2l DESTINATION lib)
install(FILES w2l.h DESTINATION include)
