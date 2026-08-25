// What this pins: claims of the form "this is the one place X happens".
//
// Four of them were found false in a week, all in the ownership and ABI code,
// all by a person reading nearby files. `ABI/EntityHeaderPrefix.h` justified a
// predicate by describing two of the three producers of the frame-ownership
// marker. `Ops/GetItemOps.cpp` called itself the only one of the three.
// `ABI/BoxLayout.cpp` called itself the one place a memref descriptor is
// built, and stayed that way after `buildMemRef1D` arrived for the exception
// triple and built the same five fields. Every one of those conclusions was
// still TRUE; only the argument had stopped covering the code. That is the
// failure mode worth a test, because nothing about it looks wrong.
//
// So a "one place" claim in this tree should either be enforced or reworded.
// This file is where the enforced ones live. Adding a claim means adding a
// case here; the alternative is writing a sentence that a later commit turns
// into fiction with nothing to say so.
//
// The marker's own case is in ModelCorrespondenceTests, because there it is
// also a statement about the gate's domain against `proof/`.

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <string>
#include <system_error>

namespace {

// Files under src/lython whose text contains `needle`.
llvm::SmallVector<std::string, 4> filesContaining(llvm::StringRef needle) {
  llvm::SmallVector<std::string, 4> hits;
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           entry(LYTHON_SOURCE_DIR "/src/lython", error),
       end;
       entry != end && !error; entry.increment(error)) {
    llvm::StringRef path = entry->path();
    if (!path.ends_with(".cpp") && !path.ends_with(".h"))
      continue;
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer)
      continue;
    if (buffer.get()->getBuffer().contains(needle))
      hits.push_back(path.str());
  }
  return hits;
}

bool named(llvm::StringRef path, llvm::StringRef suffix) {
  return path.ends_with(suffix);
}

} // namespace

// `Common/MemRef1D.h` assembles the rank-1 memref descriptor, and is the only
// thing that may.
TEST(SoleProducerTest, TheRank1DescriptorIsAssembledInOnePlace) {
  llvm::SmallVector<std::string, 4> unexpected;
  for (const std::string &path : filesContaining("InsertValueOp::create"))
    if (!named(path, "Common/MemRef1D.h"))
      unexpected.push_back(path);

  EXPECT_TRUE(unexpected.empty())
      << "an LLVM aggregate is assembled outside the one file that may: "
      << llvm::join(unexpected, ", ")
      << ". If it is a rank-1 memref descriptor, call buildMemRef1D from "
         "Common/MemRef1D.h -- one assembler is what keeps the box path and "
         "the exception triple agreeing on the five fields. If it is some "
         "other struct, add it here and say what it is.";
}

// The allow-list above is only meaningful while every entry is load-bearing.
// A stale exemption is the same species of rot as a stale comment.
TEST(SoleProducerTest, TheDescriptorAssemblerExemptionsAreStillUsed) {
  for (llvm::StringRef exemption : {"Common/MemRef1D.h"}) {
    bool used = false;
    for (const std::string &path : filesContaining("InsertValueOp::create"))
      used = used || named(path, exemption);
    EXPECT_TRUE(used) << exemption.str()
                      << " no longer assembles an aggregate; drop it from the "
                         "allow-list in this test rather than leaving an "
                         "exemption that permits something nobody meant";
  }
}
