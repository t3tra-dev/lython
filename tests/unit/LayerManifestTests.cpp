// What this pins: every test in this binary is claimed by exactly one layer of
// the ctest label taxonomy, and every layer pattern still claims something.
//
// tests/CMakeLists.txt registers this binary once per layer, each registration
// filtered to that layer's gtest patterns, so that `ctest -L fast` can be cut
// by how far into the compiler a test reaches rather than by which file it
// lives in. A test no pattern matches is registered by no layer and therefore
// runs nowhere -- it would be added, pass locally under `ctest` in the author's
// own gtest invocation, and never run again. A pattern that matches nothing is
// the same accident from the other end: renaming a suite silently drops every
// test in it. Both are failures here, by name.
//
// The manifest arrives in a header generated from the same CMake lists that
// build the discovery filters, so this test and the registration cannot
// disagree about what the layers are.

#include "LayerManifest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#ifndef LYTHON_UNIT_LAYER_MANIFEST
#error "LayerManifest.h must define LYTHON_UNIT_LAYER_MANIFEST"
#endif

namespace {

struct Layer {
  std::string name;
  std::vector<std::string> patterns;
};

// "layer=pat:pat|layer=pat:pat", the encoding tests/CMakeLists.txt generates.
std::vector<Layer> parseManifest(llvm::StringRef manifest) {
  std::vector<Layer> layers;
  llvm::SmallVector<llvm::StringRef, 8> entries;
  manifest.split(entries, '|', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef entry : entries) {
    auto [name, patterns] = entry.split('=');
    Layer layer;
    layer.name = name.str();
    llvm::SmallVector<llvm::StringRef, 8> split;
    patterns.split(split, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef pattern : split)
      layer.patterns.push_back(pattern.str());
    layers.push_back(std::move(layer));
  }
  return layers;
}

// gtest filter wildcards: '*' spans any run of characters, '?' one character.
// Matching them here rather than asking gtest keeps this test independent of
// gtest internals that are not part of its public API.
bool matchesPattern(llvm::StringRef pattern, llvm::StringRef name) {
  if (pattern.empty())
    return name.empty();
  if (pattern.front() == '*') {
    llvm::StringRef rest = pattern.drop_front();
    for (std::size_t skip = 0; skip <= name.size(); ++skip)
      if (matchesPattern(rest, name.drop_front(skip)))
        return true;
    return false;
  }
  if (name.empty())
    return false;
  if (pattern.front() != '?' && pattern.front() != name.front())
    return false;
  return matchesPattern(pattern.drop_front(), name.drop_front());
}

std::vector<std::string> registeredTestNames() {
  std::vector<std::string> names;
  const ::testing::UnitTest &unitTest = *::testing::UnitTest::GetInstance();
  // total_*_count(), unlike reportable_*_count(), counts tests the active
  // --gtest_filter excluded. Each layer's ctest registration runs this binary
  // under a filter, so only the total counts see the whole binary.
  for (int suiteIndex = 0; suiteIndex < unitTest.total_test_suite_count();
       ++suiteIndex) {
    const ::testing::TestSuite &suite = *unitTest.GetTestSuite(suiteIndex);
    for (int testIndex = 0; testIndex < suite.total_test_count(); ++testIndex)
      names.push_back(std::string(suite.name()) + "." +
                      suite.GetTestInfo(testIndex)->name());
  }
  return names;
}

TEST(LayerManifestTest, EveryTestIsClaimedByExactlyOneLayer) {
  std::vector<Layer> layers = parseManifest(LYTHON_UNIT_LAYER_MANIFEST);
  ASSERT_FALSE(layers.empty()) << "manifest parsed to nothing: "
                               << LYTHON_UNIT_LAYER_MANIFEST;
  std::vector<std::string> names = registeredTestNames();
  ASSERT_FALSE(names.empty()) << "no registered tests were found, so this "
                                 "check would pass vacuously";

  unsigned classified = 0;
  for (const std::string &name : names) {
    std::vector<std::string> claimants;
    for (const Layer &layer : layers)
      for (const std::string &pattern : layer.patterns)
        if (matchesPattern(pattern, name)) {
          claimants.push_back(layer.name + " (" + pattern + ")");
          break;
        }
    if (claimants.size() == 1) {
      ++classified;
      continue;
    }
    if (claimants.empty()) {
      ADD_FAILURE()
          << name << " is in no layer, so no ctest registration runs it.\n"
          << "Add it to a LYTHON_UNIT_LAYER_<layer> list in "
             "tests/CMakeLists.txt. Pick the layer by how far into the "
             "compiler the test reaches, not by which file it lives in: "
             "parse/emit/tables/meta are the cheap in-process layers that "
             "`ctest -L fast` runs, lower is a full lowering to LLVM IR.";
      continue;
    }
    std::string joined;
    for (const std::string &claimant : claimants)
      joined += (joined.empty() ? "" : ", ") + claimant;
    ADD_FAILURE() << name << " is claimed by " << claimants.size()
                  << " layers (" << joined
                  << "), so ctest would register it more than once under "
                     "conflicting labels. Narrow one of the patterns.";
  }
  EXPECT_EQ(classified, names.size())
      << "classified " << classified << " of " << names.size()
      << " registered tests";
}

TEST(LayerManifestTest, EveryLayerPatternClaimsSomething) {
  std::vector<Layer> layers = parseManifest(LYTHON_UNIT_LAYER_MANIFEST);
  std::vector<std::string> names = registeredTestNames();
  for (const Layer &layer : layers) {
    EXPECT_FALSE(layer.patterns.empty()) << "layer " << layer.name
                                         << " has no patterns";
    for (const std::string &pattern : layer.patterns) {
      unsigned matches = 0;
      for (const std::string &name : names)
        matches += matchesPattern(pattern, name) ? 1 : 0;
      EXPECT_GT(matches, 0u)
          << "layer " << layer.name << " pattern '" << pattern
          << "' matches none of the " << names.size()
          << " registered tests. A renamed or deleted suite leaves a pattern "
             "like this behind, and every test it used to cover stops running "
             "without any test going red.";
    }
  }
}

} // namespace
