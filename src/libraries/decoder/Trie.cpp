/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <functional>

#include "libraries/decoder/Trie.h"

namespace w2l {

const double kMinusLogThreshold = -39.14;

const TrieNode* Trie::getRoot() const {
  return root_.get();
}

TrieNodePtr
Trie::insert(const std::vector<int>& indices, int label, float score) {
  TrieNodePtr node = root_;
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    if (idx < 0 || idx >= maxChildren_) {
      throw std::out_of_range(
          "[Trie] Invalid letter index: " + std::to_string(idx));
    }
    if (node->children.find(idx) == node->children.end()) {
      node->children[idx] = std::make_shared<TrieNode>(idx);
    }
    node = node->children[idx];
  }
  if (node->labels.size() < kTrieMaxLabel) {
    node->labels.push_back(label);
    node->scores.push_back(score);
  } else {
    std::cerr << "[Trie] Trie label number reached limit: " << kTrieMaxLabel
              << "\n";
  }
  return node;
}

TrieNodePtr Trie::search(const std::vector<int>& indices) {
  TrieNodePtr node = root_;
  for (auto idx : indices) {
    if (idx < 0 || idx >= maxChildren_) {
      throw std::out_of_range(
          "[Trie] Invalid letter index: " + std::to_string(idx));
    }
    if (node->children.find(idx) == node->children.end()) {
      return nullptr;
    }
    node = node->children[idx];
  }
  return node;
}

/* logadd */
double TrieLogAdd(double log_a, double log_b) {
  double minusdif;
  if (log_a < log_b) {
    std::swap(log_a, log_b);
  }
  minusdif = log_b - log_a;
  if (minusdif < kMinusLogThreshold) {
    return log_a;
  } else {
    return log_a + log1p(exp(minusdif));
  }
}

void smearNode(TrieNodePtr node, SmearingMode smearMode) {
  node->maxScore = -std::numeric_limits<float>::infinity();
  for (auto score : node->scores) {
    node->maxScore = TrieLogAdd(node->maxScore, score);
  }
  for (auto child : node->children) {
    auto childNode = child.second;
    smearNode(childNode, smearMode);
    if (smearMode == SmearingMode::LOGADD) {
      node->maxScore = TrieLogAdd(node->maxScore, childNode->maxScore);
    } else if (
        smearMode == SmearingMode::MAX &&
        childNode->maxScore > node->maxScore) {
      node->maxScore = childNode->maxScore;
    }
  }
}

void Trie::smear(SmearingMode smearMode) {
  if (smearMode != SmearingMode::NONE) {
    smearNode(root_, smearMode);
  }
}

FlatTrie toFlatTrie(const TrieNode *root) {
    std::vector<int32_t> out;
    std::unordered_map<const TrieNode *, size_t> zeroOffsets;

    // All fields have size 4; number of fields needed
    auto flatTrieNodeN = [](const TrieNode *node)
    {
        return 4 + node->children.size() + node->labels.size();
    };

    std::function<void (const TrieNode *)> allocAndMap;
    size_t allocSize = 0;
    allocAndMap = [&](const TrieNode *node) {
        zeroOffsets[node] = allocSize;
        allocSize += flatTrieNodeN(node);

        std::vector<std::pair<int, std::shared_ptr<TrieNode>>> children(node->children.begin(), node->children.end());
        std::stable_sort(children.begin(), children.end(), [](auto &i, auto &j) {
            return i.first < j.first;
        });
        for (auto &child : children)
            allocAndMap(child.second.get());
    };
    allocAndMap(root);
    out.resize(allocSize);

    std::function<void (const TrieNode *)> writeData;
    writeData = [&](const TrieNode *node) {
        auto thisOffset = zeroOffsets[node];
        auto flat = reinterpret_cast<FlatTrieNode *>(&out[thisOffset]);
        flat->idx = node->idx;
        flat->maxScore = node->maxScore;
        flat->nChildren = node->children.size();
        flat->nLabel = node->labels.size();

        std::vector<std::pair<int, std::shared_ptr<TrieNode>>> children(node->children.begin(), node->children.end());
        std::stable_sort(children.begin(), children.end(), [](auto &i, auto &j) {
            return i.first < j.first;
        });
        int iChild = 0;
        for (const auto &pair : children) {
            flat->data[iChild++] = zeroOffsets[pair.second.get()] - thisOffset;
        }
        for (int i = 0; i < node->labels.size(); ++i) {
            flat->data[iChild + i] = node->labels[i];
        }
        std::sort(flat->data + iChild, flat->data + iChild + node->labels.size());
        for (const auto &child : children)
            writeData(child.second.get());
    };
    writeData(root);

    return FlatTrie{std::move(out)};
}

} // namespace w2l
