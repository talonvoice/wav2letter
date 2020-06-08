/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace w2l {

constexpr int kTrieMaxLabel = 6;

enum class SmearingMode {
  NONE = 0,
  MAX = 1,
  LOGADD = 2,
};

/**
 * TrieNode is the trie node structure in Trie.
 */
struct TrieNode {
  explicit TrieNode(int idx)
      : children(std::unordered_map<int, std::shared_ptr<TrieNode>>()),
        idx(idx),
        maxScore(0) {
    labels.reserve(kTrieMaxLabel);
    scores.reserve(kTrieMaxLabel);
  }

  // Pointers to the childern of a node
  std::unordered_map<int, std::shared_ptr<TrieNode>> children;

  // Node index
  int idx;

  // Labels of words that are constructed from the given path. Note that
  // `labels` is nonempty only if the current node represents a completed token.
  std::vector<int> labels;

  // Scores (`scores` should have the same size as `labels`)
  std::vector<float> scores;

  // Maximum score of all the labels if this node is a leaf,
  // otherwise it will be the value after trie smearing.
  float maxScore;
};

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */
struct FlatTrieNode
{
    int32_t idx;
    float maxScore;
    int32_t nChildren;
    int32_t nLabel;

    // nChildren children offsets
    // nLabel labels
    int32_t data[1];

    const FlatTrieNode *child(size_t i) const {
        return reinterpret_cast<const FlatTrieNode *>(
            reinterpret_cast<const int32_t *>(this) + data[i]);
    }

    int32_t label(size_t i) const {
        return data[nChildren + i];
    }

    const FlatTrieNode *find(size_t token) const {
        // binary search children for token (requires ordered flattrie)
        int L = 0;
        int R = nChildren - 1;
        while (L <= R) {
            int mid = (L + R) / 2;
            auto child = this->child(mid);
            if (child->idx < token) {
                L = mid + 1;
            } else if (child->idx == token) {
                return child;
            } else {
                R = mid - 1;
            }
        }
        return nullptr;

        // non-binary search
        /*
        for (ssize_t i = 0; i < nChildren; i++) {
            auto child = this->child(i);
            if (child->idx == token) {
                return child;
            }
        }
        return nullptr;
        */
    }
};
#pragma pack(pop)

/**
 * A flattenend form of the Trie that packs data tightly and
 * works with relative offsets. Storage can be written to disk
 * and restored as one chunk.
 */
struct FlatTrie
{
    std::vector<int32_t> storage;
    const FlatTrieNode *getRoot() const {
        return reinterpret_cast<const FlatTrieNode *>(storage.data());
    }
};
using FlatTriePtr = std::shared_ptr<FlatTrie>;

/**
 * Converts a regular trie to the flattened form.
 */
FlatTrie toFlatTrie(const TrieNode *trie);

using TrieNodePtr = std::shared_ptr<TrieNode>;

/**
 * Trie is used to store the lexicon in langiage model. We use it to limit
 * the search space in deocder and quickly look up scores for a given token
 * (completed word) or make prediction for incompleted ones based on smearing.
 */
class Trie {
 public:
  Trie(int maxChildren, int rootIdx)
      : root_(std::make_shared<TrieNode>(rootIdx)), maxChildren_(maxChildren) {}

  /* Return the root node pointer */
  const TrieNode* getRoot() const;

  /* Insert a token into trie with label */
  TrieNodePtr insert(const std::vector<int>& indices, int label, float score);

  /* Get the labels for a given token */
  TrieNodePtr search(const std::vector<int>& indices);

  /**
   * Smearing the trie using the valid labels inserted in the trie so as to get
   * score on each node (incompleted token).
   * For example, if smear_mode is MAX, then for node "a" in path "c"->"a", we
   * will select the maximum score from all its children like "c"->"a"->"t",
   * "c"->"a"->"n", "c"->"a"->"r"->"e" and so on.
   * This process will be carry out recusively on all the nodes.
   */
  void smear(const SmearingMode smear_mode);

 private:
  TrieNodePtr root_;
  int maxChildren_; // The maximum number of childern for each node. It is
                    // usually the size of letters or phonmes.
};

using TriePtr = std::shared_ptr<Trie>;

} // namespace w2l
