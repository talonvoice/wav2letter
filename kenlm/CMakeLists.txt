cmake_minimum_required(VERSION 2.6)
project(kenlm)

set(KENLM_LM_SOURCE
    lm/bhiksha.cc
    lm/binary_format.cc
    lm/config.cc
    lm/lm_exception.cc
    lm/model.cc
    lm/quantize.cc
    lm/read_arpa.cc
    lm/search_hashed.cc
    lm/search_trie.cc
    lm/sizes.cc
    lm/trie.cc
    lm/trie_sort.cc
    lm/value_build.cc
    lm/virtual_interface.cc
    lm/vocab.cc

    util/bit_packing.cc
    util/ersatz_progress.cc
    util/exception.cc
    util/file.cc
    util/file_piece.cc
    util/float_to_string.cc
    util/integer_to_string.cc
    util/mmap.cc
    util/murmur_hash.cc
    util/parallel_read.cc
    util/pool.cc
    util/read_compressed.cc
    util/scoped.cc
    util/spaces.cc
    util/string_piece.cc
    util/usage.cc

    util/double-conversion/bignum-dtoa.cc
    util/double-conversion/bignum.cc
    util/double-conversion/cached-powers.cc
    util/double-conversion/diy-fp.cc
    util/double-conversion/double-conversion.cc
    util/double-conversion/fast-dtoa.cc
    util/double-conversion/fixed-dtoa.cc
    util/double-conversion/strtod.cc
)

set(KENLM_LIBRARIES kenlm PARENT_SCOPE)
set(KENLM_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} PARENT_SCOPE)
set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum supported ngram order")
include_directories(${PROJECT_SOURCE_DIR})

if (NOT MSVC)
    set(THREADS pthread)
endif()

add_library(kenlm STATIC ${KENLM_LM_SOURCE})
target_link_libraries(kenlm ${THREADS})
target_compile_definitions(kenlm PUBLIC -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})
