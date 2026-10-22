#include "Common/PythonSourceRange.h"
#include "Common/RuntimeSupport.h"
#include "Mangling.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <string>

// The frame pass: writes the live call stack `_frame` reads
// (lowering/Common/FrameSupportBuilder.cpp), in the functions that can reach a
// read and nowhere else.

namespace py {
namespace {

bool isFrameRead(llvm::StringRef callee) {
  return callee == "LyFrame_Size" || callee == "LyFrame_Line" ||
         callee == "LyFrame_TextLen" || callee == "LyFrame_TextCopy" ||
         callee.starts_with("LyFrameMod_");
}

// What CPython names the frame: the definition's own name, `<module>` for the
// module body, the clone family's root for a generator or primitive clone.
std::string frameName(mlir::LLVM::LLVMFuncOp function) {
  if (auto python = function->getAttrOfType<mlir::StringAttr>(
          mangling::kFunctionNameAttr))
    return python.getValue().str();
  llvm::StringRef symbol = function.getSymName();
  if (symbol == "__main__")
    return "<module>";
  for (llvm::StringRef marker : {"__lyrt_prim", "__lyrt_gen"}) {
    std::size_t suffix = symbol.find(marker);
    if (suffix != llvm::StringRef::npos)
      symbol = symbol.take_front(suffix);
  }
  return symbol.str();
}

// The module a definition belongs to: its mangled symbol's components less
// those its qualified name spells (common/Mangling.h).
std::optional<std::string> definitionModule(mlir::LLVM::LLVMFuncOp function) {
  auto qualname = function->getAttrOfType<mlir::StringAttr>(
      mangling::kFunctionQualnameAttr);
  llvm::StringRef symbol = function.getSymName();
  if (!qualname || !symbol.consume_front("_ZN"))
    return std::nullopt;
  llvm::SmallVector<llvm::StringRef, 4> components;
  while (!symbol.empty() && symbol.front() != 'E') {
    std::size_t length = 0;
    if (symbol.consumeInteger(10, length) || length > symbol.size())
      return std::nullopt;
    components.push_back(symbol.take_front(length));
    symbol = symbol.drop_front(length);
  }
  llvm::SmallVector<llvm::StringRef, 4> qualified;
  qualname.getValue().split(qualified, '.');
  if (qualified.size() >= components.size())
    return std::nullopt;
  std::string module;
  for (llvm::StringRef component :
       llvm::ArrayRef(components).drop_back(qualified.size())) {
    if (!module.empty())
      module += '.';
    module += component.str();
  }
  return module;
}

struct FrameRecord {
  std::string file;
  std::string name;
  std::int32_t line = 0;
};

class FrameWriter {
public:
  explicit FrameWriter(mlir::ModuleOp module)
      : module(module), builder(module.getContext()) {}

  mlir::Value text(mlir::Location loc, llvm::StringRef value) {
    auto &symbol = strings[value];
    if (symbol.empty()) {
      symbol = "py.frame.text." + std::to_string(strings.size());
      std::string data = value.str();
      data.push_back('\0');
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      mlir::LLVM::GlobalOp::create(
          builder, builder.getUnknownLoc(),
          mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), data.size()),
          /*isConstant=*/true, mlir::LLVM::Linkage::Internal, symbol,
          builder.getStringAttr(data), /*alignment=*/1);
    }
    return mlir::LLVM::AddressOfOp::create(
        builder, loc, mlir::LLVM::LLVMPointerType::get(module.getContext()),
        symbol);
  }

  mlir::OpBuilder &at(mlir::Operation *op) {
    builder.setInsertionPoint(op);
    return builder;
  }

private:
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  llvm::StringMap<std::string> strings;
};

} // namespace

void installFrameRecords(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 64> functions;
  llvm::StringMap<mlir::LLVM::LLVMFuncOp> bySymbol;
  module.walk([&](mlir::LLVM::LLVMFuncOp function) {
    functions.push_back(function);
    bySymbol[function.getSymName()] = function;
  });

  // ⭐ WHO CAN REACH A READ. A direct call reaches what its callee reaches;
  // an indirect call reaches whatever a function whose address is taken
  // reaches, so it counts once any such function does.
  // ⛔ A call into a runtime DECLARATION does not count, though the runtime
  // calls back into Python (a `key=` function): the declaration's body is not
  // here to read. The callback's frame lands on its caller's caller, and the
  // function that handed it over is missing from the stack, as a C function
  // is missing from CPython's.
  llvm::DenseSet<mlir::Operation *> addressTaken;
  module.walk([&](mlir::LLVM::AddressOfOp address) {
    if (auto found = bySymbol.find(address.getGlobalName());
        found != bySymbol.end())
      addressTaken.insert(found->second);
  });
  llvm::DenseSet<mlir::Operation *> reaches;
  bool indirectReaches = false;
  auto callReaches = [&](mlir::LLVM::CallOp call) {
    std::optional<llvm::StringRef> callee = call.getCallee();
    if (!callee)
      return indirectReaches;
    if (isFrameRead(*callee))
      return true;
    auto found = bySymbol.find(*callee);
    return found != bySymbol.end() && reaches.contains(found->second);
  };
  bool anyRead = false;
  module.walk([&](mlir::LLVM::CallOp call) {
    if (std::optional<llvm::StringRef> callee = call.getCallee())
      anyRead |= isFrameRead(*callee);
  });
  if (!anyRead)
    return;
  for (bool changed = true; changed;) {
    changed = false;
    for (mlir::LLVM::LLVMFuncOp function : functions) {
      if (reaches.contains(function) || function.isExternal())
        continue;
      bool reached = false;
      function.walk([&](mlir::LLVM::CallOp call) {
        if (!reached && callReaches(call))
          reached = true;
      });
      if (!reached)
        continue;
      reaches.insert(function);
      changed = true;
      if (addressTaken.contains(function))
        indirectReaches = true;
    }
  }

  // The module of each file, read off the definitions written in it. A frame
  // names its file, and an inlined body's frame names nothing else.
  llvm::StringMap<std::string> moduleOfFile;
  for (mlir::LLVM::LLVMFuncOp function : functions) {
    std::optional<mlir::FileLineColLoc> source =
        findPythonSourceLoc(function.getLoc());
    if (!source)
      continue;
    llvm::StringRef file = source->getFilename().getValue();
    if (function.getSymName() == "__main__")
      moduleOfFile[file] = "__main__";
    else if (std::optional<std::string> owner = definitionModule(function))
      moduleOfFile.try_emplace(file, *owner);
  }

  FrameWriter writer(module);
  mlir::MLIRContext *context = module.getContext();
  auto i64 = mlir::IntegerType::get(context, 64);
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  for (mlir::LLVM::LLVMFuncOp function : functions) {
    if (!reaches.contains(function) || function.isExternal())
      continue;
    std::optional<mlir::FileLineColLoc> own =
        findPythonSourceLoc(function.getLoc());
    if (!own)
      continue;
    std::string ownName = frameName(function);
    llvm::SmallVector<mlir::LLVM::CallOp, 8> sites;
    function.walk([&](mlir::LLVM::CallOp call) {
      if (callReaches(call))
        sites.push_back(call);
    });
    if (sites.empty())
      continue;
    mlir::Block &entry = function.getBody().front();
    mlir::OpBuilder &entryBuilder = writer.at(&entry.front());
    mlir::Value depth =
        mlir::LLVM::CallOp::create(entryBuilder, function.getLoc(),
                                   mlir::TypeRange{i64}, "LyFrame_Size",
                                   mlir::ValueRange{})
            .getResult();
    for (mlir::LLVM::CallOp call : sites) {
      // Innermost first, as the traceback's frames are (PythonSourceRange.h).
      llvm::SmallVector<FrameRecord, 2> frames;
      if (std::optional<PythonSourceRange> range =
              pythonSourceRange(call.getLoc())) {
        frames.push_back({range->filename,
                          range->functionName.empty() ? ownName
                                                      : range->functionName,
                          range->line});
        for (const PythonInlineFrame &outer : range->inlinedAt)
          frames.push_back(
              {outer.filename.empty() ? range->filename : outer.filename,
               outer.functionName.empty() ? ownName : outer.functionName,
               outer.line});
      } else {
        // A call the lowering wrote with no place in the source still runs in
        // this function; its frame is the definition's line.
        frames.push_back({own->getFilename().getValue().str(), ownName,
                          static_cast<std::int32_t>(own->getLine())});
      }
      mlir::OpBuilder &b = writer.at(call);
      mlir::Location loc = call.getLoc();
      std::int64_t level = 0;
      for (const FrameRecord &frame : llvm::reverse(frames)) {
        mlir::Value index = depth;
        if (level != 0)
          index = mlir::LLVM::AddOp::create(
              b, loc, depth,
              mlir::LLVM::ConstantOp::create(b, loc, i64, level));
        auto owning = moduleOfFile.find(frame.file);
        mlir::Value file = writer.text(loc, frame.file);
        mlir::Value name = writer.text(loc, frame.name);
        mlir::Value owner = writer.text(
            loc, owning == moduleOfFile.end() ? "" : owning->second);
        mlir::Value line =
            mlir::LLVM::ConstantOp::create(b, loc, i64, frame.line);
        mlir::LLVM::CallOp::create(b, loc, mlir::TypeRange{}, "LyFrame_Set",
                                   mlir::ValueRange{index, file, name, owner,
                                                    line});
        ++level;
      }
    }
  }
  if (!module.lookupSymbol("LyFrame_Size")) {
    mlir::OpBuilder builder(context);
    builder.setInsertionPointToEnd(module.getBody());
    mlir::LLVM::LLVMFuncOp::create(
        builder, builder.getUnknownLoc(), "LyFrame_Size",
        mlir::LLVM::LLVMFunctionType::get(i64, {}));
  }
  if (!module.lookupSymbol("LyFrame_Set")) {
    mlir::OpBuilder builder(context);
    builder.setInsertionPointToEnd(module.getBody());
    mlir::LLVM::LLVMFuncOp::create(
        builder, builder.getUnknownLoc(), "LyFrame_Set",
        mlir::LLVM::LLVMFunctionType::get(
            mlir::LLVM::LLVMVoidType::get(context),
            {i64, ptr, ptr, ptr, i64}));
  }
}

} // namespace py
