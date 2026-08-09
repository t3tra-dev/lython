// What this pins: the compiler's ownership vocabulary cannot grow past the
// model without someone saying what the new word means.
//
// `proof/` is the design, and the whole connection between it and the C++ was
// eight comments. Three divergences in one week -- the exception chain node
// holding descriptors as words, a module global's cell holding an address, the
// `except*` frame handle typed `i64` -- were each found by a person reading
// code. None produced a diagnostic.
//
// The map is the comment block at the top of common/Ownership.h, beside the
// declarations it maps. A map nothing reads is prose, and prose does not fail
// a build: the same finding as RuntimeRaisePathTests', reached from the other
// end. So this reads both lists out of that one file and compares them.
//
// WHAT IT CANNOT DO, said here so the coverage is not overread: it checks that
// every attribute is ACCOUNTED FOR, not that the accounting is right. Only a
// person can decide that `aggregate_retain` is `setField`. What a machine can
// refuse is a seventeenth attribute with no row, and a model that grows a step
// while the block still claims the old count. That second half has already
// earned itself once: `callOut` landed, `Instr` went from eight to nine, and
// this is what said the map had to be revisited.

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>

namespace {

std::string readOrDie(const std::string &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  EXPECT_TRUE(static_cast<bool>(buffer))
      << "cannot read " << path
      << "; this test compares the tree against itself and needs it";
  if (!buffer)
    return {};
  return buffer.get()->getBuffer().str();
}

llvm::SmallVector<llvm::StringRef, 256> linesOf(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef, 256> lines;
  text.split(lines, '\n');
  return lines;
}

// The `ly.ownership.<name>` on one line, or empty. A bare `ly.ownership.` in
// prose is the prefix, not a name.
llvm::StringRef attributeOn(llvm::StringRef line) {
  const llvm::StringRef prefix = "ly.ownership.";
  std::size_t at = line.find(prefix);
  if (at == llvm::StringRef::npos)
    return {};
  std::size_t end = at + prefix.size();
  while (
      end < line.size() &&
      (std::isalnum(static_cast<unsigned char>(line[end])) || line[end] == '_'))
    ++end;
  if (end == at + prefix.size())
    return {};
  return line.substr(at, end - at);
}

// Both lists live in one file, so they are told apart by their shape: a
// declaration QUOTES the attribute (and wraps it onto its own line, which is
// why this cannot key on `StringLiteral` being on the same line); a row is a
// comment and quotes nothing.
struct Vocabulary {
  llvm::StringSet<> declared;
  llvm::StringSet<> mapped;
};

Vocabulary readVocabulary(llvm::StringRef text) {
  Vocabulary vocabulary;
  for (llvm::StringRef line : linesOf(text)) {
    llvm::StringRef name = attributeOn(line);
    if (name.empty())
      continue;
    std::size_t at = line.find(name);
    bool quoted = at > 0 && line[at - 1] == '"';
    if (quoted)
      vocabulary.declared.insert(name);
    else if (line.ltrim().starts_with("//"))
      vocabulary.mapped.insert(name);
  }
  return vocabulary;
}

// Constructors of one Agda `data` declaration, counted by their `name :` lines.
unsigned constructorCount(llvm::StringRef text, llvm::StringRef declaration) {
  std::size_t at = text.find(declaration);
  EXPECT_NE(at, llvm::StringRef::npos)
      << "the model no longer declares `" << declaration.str()
      << "`; the map in common/Ownership.h is written around it";
  if (at == llvm::StringRef::npos)
    return 0;
  unsigned count = 0;
  for (llvm::StringRef line : linesOf(text.substr(at + declaration.size()))) {
    llvm::StringRef body = line.ltrim();
    if (body.empty())
      continue;
    // An unindented line ends the declaration's block.
    if (!line.starts_with("  "))
      break;
    if (body.starts_with("--"))
      continue;
    if (body.contains(" : "))
      ++count;
  }
  return count;
}

const char *kOwnershipHeader =
    LYTHON_SOURCE_DIR "/src/lython/common/Ownership.h";

} // namespace

TEST(ModelCorrespondenceTest, EveryOwnershipAttributeHasARow) {
  Vocabulary vocabulary = readVocabulary(readOrDie(kOwnershipHeader));

  ASSERT_FALSE(vocabulary.declared.empty())
      << "no attribute declarations found in common/Ownership.h";
  ASSERT_FALSE(vocabulary.mapped.empty())
      << "no rows found in the map at the top of common/Ownership.h";

  for (const auto &entry : vocabulary.declared)
    EXPECT_TRUE(vocabulary.mapped.contains(entry.getKey()))
        << entry.getKey().str()
        << " is declared in common/Ownership.h and has no row in the map at "
           "the top of that file. Say what it corresponds to in proof/, or "
           "that it corresponds to nothing and why -- the six call-boundary "
           "attributes are listed that way.";

  for (const auto &entry : vocabulary.mapped)
    EXPECT_TRUE(vocabulary.declared.contains(entry.getKey()))
        << entry.getKey().str()
        << " has a row in the map and is not declared below it; the map is "
           "describing something that is gone.";
}

// The map's claim about the model's size, checked against the model.
//
// This is the half that matters for what comes next. The map is written against
// a specific instruction set. Every attribute is mapped now, so a new
// constructor means the map has a claim to re-examine rather than a gap to
// fill -- and failing is still how it gets said.
TEST(ModelCorrespondenceTest, TheModelHasTheStepsTheMapAssumes) {
  std::string syntax =
      readOrDie(LYTHON_SOURCE_DIR "/proof/src/Proof/Program/Syntax.agda");

  EXPECT_EQ(constructorCount(syntax, "data Instr : Set where"), 10u)
      << "the model's instruction set changed; the map in common/Ownership.h "
         "places the compiler's attributes against exactly the ten it had";
  EXPECT_EQ(constructorCount(syntax, "data Term : Set where"), 5u)
      << "the model's terminators changed; the map explains that `invoke` is "
         "an unwind edge and not a call, which is a claim about this set";
}

// The mint-site count, which is the other thing the comments kept getting
// wrong.
//
// `ABI/EntityHeaderPrefix.h` justified a predicate by describing two producers
// of `ly.ownership.owned_local_object`; `Ops/GetItemOps.cpp` called itself the
// only one. There were three, and both conclusions happened to survive it.
// They are one now, and `Core/OwnedLocalMarker.h` says so -- a claim that
// needs a machine behind it, since the two it replaces did not have one.
//
// It matters to the gate and not just to tidiness:
// `verifyInitialisationWindowIn` reads a marker as the model's `dup` and tells
// it apart from `alloc` by its being a rooting cast. A fourth producer that
// marked something else would leave that domain without any test noticing.
TEST(ModelCorrespondenceTest, TheOwnedLocalMarkerHasOneMintSite) {
  const llvm::StringRef mint =
      "src/lython/lowering/Passes/Runtime/Core/OwnedLocalMarker.h";
  llvm::SmallVector<std::string, 4> writers;
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           entry(LYTHON_SOURCE_DIR "/src/lython", error),
       end;
       entry != end && !error; entry.increment(error)) {
    llvm::StringRef path = entry->path();
    if (!path.ends_with(".cpp") && !path.ends_with(".h"))
      continue;
    for (llvm::StringRef line : linesOf(readOrDie(path.str()))) {
      if (!line.contains("setAttr("))
        continue;
      if (!line.contains("kOwnedLocalObjectAttr") &&
          !line.contains("\"ly.ownership.owned_local_object\""))
        continue;
      if (path.contains(mint))
        continue;
      writers.push_back(path.str());
    }
  }

  EXPECT_TRUE(writers.empty())
      << "the frame-ownership marker is written outside " << mint.str() << ": "
      << llvm::join(writers, ", ")
      << ". Call mintOwnedLocalMarker instead. The gate that reads this "
         "attribute distinguishes the model's `dup` from its `alloc` by the "
         "marker being a rooting cast, and a producer that marks anything "
         "else drops out of what it can judge.";
}
