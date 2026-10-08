#include "Common/TypeObjects.h"

#include "Contracts.h"
#include "ExceptionTaxonomy.h"
#include "Native.h"
#include "RuntimeClasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/TargetParser/Triple.h"

namespace py::type_objects {

namespace {

constexpr llvm::StringLiteral kClassWordOfAttr{"ly.class_of"};
constexpr llvm::StringLiteral kClassWordStrideAttr{"ly.class_stride"};
// The slots each class's type object holds: class -> {slot word -> function}.
constexpr llvm::StringLiteral kSlotsAttr{"ly.type_slots"};
// A static object whose address is an identity the program compares (None):
// one object however many modules carry it.
constexpr llvm::StringLiteral kSharedStaticAttr{"ly.static.shared"};
constexpr std::int64_t kImmortal = 0x7FFFFFFFFFFFFFFF;

mlir::Type declarationType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(context, 64),
                                        kWordCount);
}

mlir::ModuleOp enclosingModule(mlir::OpBuilder &builder) {
  mlir::Block *block = builder.getInsertionBlock();
  mlir::Operation *parent = block ? block->getParentOp() : nullptr;
  if (auto module = mlir::dyn_cast_if_present<mlir::ModuleOp>(parent))
    return module;
  return parent ? parent->getParentOfType<mlir::ModuleOp>() : nullptr;
}

// ⛔ Not `GlobalOp::isDeclaration`: that is the symbol interface's default,
// false for every LLVM global; a declaration is one with no value to give.
bool declaresOnly(mlir::LLVM::GlobalOp global) {
  return !global.getValueOrNull() && global.getInitializerRegion().empty();
}

std::string nameSymbolFor(llvm::StringRef qualifiedName) {
  return ("__ly_type_name." + qualifiedName).str();
}

std::optional<Definition> recordedSource(mlir::ModuleOp module,
                                         llvm::StringRef qualifiedName) {
  auto recorded = module->getAttrOfType<mlir::ArrayAttr>(kSourceTypesAttr);
  if (!recorded)
    return std::nullopt;
  for (mlir::Attribute entry : recorded) {
    auto fields = mlir::dyn_cast<mlir::ArrayAttr>(entry);
    if (!fields || fields.size() < 3)
      continue;
    auto text = [&](unsigned index) {
      auto attr = mlir::dyn_cast<mlir::StringAttr>(fields[index]);
      return attr ? attr.getValue().str() : std::string();
    };
    if (text(0) != qualifiedName)
      continue;
    Definition definition{text(0), text(1), text(2), {}};
    for (unsigned index = 3; index < fields.size(); ++index)
      definition.mro.push_back(text(index));
    return definition;
  }
  return std::nullopt;
}

mlir::LogicalResult defineOne(mlir::ModuleOp module, const Definition &definition,
                              unsigned pointerBits) {
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::Location loc = module.getLoc();
  std::string symbol = symbolFor(definition.qualifiedName);
  if (auto existing = module.lookupSymbol<mlir::LLVM::GlobalOp>(symbol)) {
    if (!declaresOnly(existing))
      return mlir::success();
    existing.erase();
  } else if (module.lookupSymbol(symbol)) {
    return module.emitError() << "'" << symbol
                              << "' names something that is not a type object";
  }

  std::string nameSymbol = nameSymbolFor(definition.qualifiedName);
  builder.setInsertionPointToStart(module.getBody());
  if (!module.lookupSymbol(nameSymbol)) {
    std::string text = definition.name + '\0';
    mlir::LLVM::GlobalOp::create(
        builder, loc,
        mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), text.size()),
        /*isConstant=*/true, mlir::LLVM::Linkage::LinkonceODR, nameSymbol,
        builder.getStringAttr(text));
  }
  auto global = mlir::LLVM::GlobalOp::create(
      builder, loc, declarationType(context), /*isConstant=*/true,
      mlir::LLVM::Linkage::LinkonceODR, symbol, mlir::Attribute(),
      /*alignment=*/16);

  llvm::SmallVector<ImageField, 8> fields;
  appendImageWord(fields, kImmortal);
  appendImageWord(fields, 0);
  if (definition.base.empty()) {
    appendImageWord(fields, 0);
  } else {
    declare(module, definition.base);
    appendImageAddress(fields, symbolFor(definition.base));
  }
  appendImageAddress(fields, nameSymbol);
  appendImageWord(fields, static_cast<std::int64_t>(definition.name.size()));
  std::string mroSymbol = ("__ly_type_mro." + definition.qualifiedName);
  appendImageAddress(fields, mroSymbol);
  mlir::DictionaryAttr slots;
  if (auto table = module->getAttrOfType<mlir::DictionaryAttr>(kSlotsAttr))
    slots = table.getAs<mlir::DictionaryAttr>(definition.qualifiedName);
  for (const SlotHook &hook : kSlotHooks) {
    auto slot = slots ? slots.getAs<mlir::StringAttr>(std::to_string(hook.word))
                      : mlir::StringAttr();
    // ⛔ Not for a hook nothing calls. Its slots would be the only thing
    // naming every arm -- str's repr with its escape tables in a program
    // that prints one str -- and a slot no hook reads is never called.
    auto caller = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(hook.name);
    if (slot && caller && !caller.isExternal() &&
        module.lookupSymbol(slot.getValue()))
      appendImageAddress(fields, slot.getValue());
    else
      appendImageWord(fields, 0);
  }
  setImage(builder, loc, global, fields, pointerBits);

  builder.setInsertionPointToStart(module.getBody());
  auto mro = mlir::LLVM::GlobalOp::create(
      builder, loc, declarationType(context), /*isConstant=*/true,
      mlir::LLVM::Linkage::LinkonceODR, mroSymbol, mlir::Attribute(),
      /*alignment=*/8);
  llvm::SmallVector<ImageField, 8> entries;
  appendImageAddress(entries, symbol);
  for (const std::string &ancestor : definition.mro) {
    declare(module, ancestor);
    appendImageAddress(entries, symbolFor(ancestor));
  }
  appendImageWord(entries, 0);
  setImage(builder, loc, mro, entries, pointerBits);
  return mlir::success();
}

} // namespace

void appendImageWord(llvm::SmallVectorImpl<ImageField> &fields,
                     std::int64_t word) {
  std::int8_t bytes[8];
  for (unsigned byte = 0; byte < 8; ++byte)
    bytes[byte] = static_cast<std::int8_t>(
        (static_cast<std::uint64_t>(word) >> (8 * byte)) & 0xff);
  appendImageBytes(fields, bytes);
}

void appendImageBytes(llvm::SmallVectorImpl<ImageField> &fields,
                      llvm::ArrayRef<std::int8_t> bytes) {
  if (bytes.empty())
    return;
  if (fields.empty() || !fields.back().symbol.empty())
    fields.emplace_back();
  fields.back().bytes.append(bytes.begin(), bytes.end());
}

void appendImageAddress(llvm::SmallVectorImpl<ImageField> &fields,
                        llvm::StringRef symbol, std::int64_t offset) {
  ImageField field;
  field.symbol = symbol.str();
  field.offset = offset;
  fields.push_back(std::move(field));
}

void setImage(mlir::OpBuilder &builder, mlir::Location loc,
              mlir::LLVM::GlobalOp global, llvm::ArrayRef<ImageField> fields,
              unsigned pointerBits) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Type i8 = builder.getI8Type();
  mlir::Type i32 = builder.getI32Type();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  bool narrow = pointerBits == 32;
  llvm::SmallVector<mlir::Type, 16> fieldTypes;
  for (const ImageField &field : fields) {
    if (!field.symbol.empty()) {
      fieldTypes.push_back(ptrType);
      if (narrow)
        fieldTypes.push_back(i32);
      continue;
    }
    fieldTypes.push_back(mlir::LLVM::LLVMArrayType::get(i8, field.bytes.size()));
  }
  auto imageType =
      mlir::LLVM::LLVMStructType::getLiteral(context, fieldTypes,
                                             /*isPacked=*/true);
  global.setGlobalTypeAttr(mlir::TypeAttr::get(imageType));
  global.removeValueAttr();
  mlir::Region &region = global.getInitializerRegion();
  region.getBlocks().clear();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&region);
  mlir::Value image =
      mlir::LLVM::UndefOp::create(builder, loc, imageType).getResult();
  std::int64_t position = 0;
  for (const ImageField &field : fields) {
    auto insert = [&](mlir::Value value) {
      image = mlir::LLVM::InsertValueOp::create(
          builder, loc, image, value, llvm::ArrayRef<std::int64_t>{position++});
    };
    if (!field.symbol.empty()) {
      mlir::Value address = mlir::LLVM::AddressOfOp::create(
          builder, loc, ptrType, field.symbol);
      if (field.offset != 0)
        address = mlir::LLVM::GEPOp::create(
            builder, loc, ptrType, i8, address,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{
                static_cast<std::int32_t>(field.offset)});
      insert(address);
      if (narrow)
        insert(mlir::LLVM::ConstantOp::create(builder, loc, i32,
                                              builder.getI32IntegerAttr(0)));
      continue;
    }
    auto bytesType = mlir::RankedTensorType::get(
        {static_cast<std::int64_t>(field.bytes.size())}, i8);
    mlir::Type fieldType = fieldTypes[position];
    insert(mlir::LLVM::ConstantOp::create(
        builder, loc, fieldType,
        mlir::DenseElementsAttr::get(bytesType,
                                     llvm::ArrayRef<std::int8_t>(field.bytes))));
  }
  mlir::LLVM::ReturnOp::create(builder, loc, image);
}

std::string symbolFor(llvm::StringRef qualifiedName) {
  return (kSymbolPrefix + qualifiedName).str();
}

std::optional<llvm::StringRef> qualifiedNameOf(llvm::StringRef symbol) {
  if (!symbol.consume_front(kSymbolPrefix))
    return std::nullopt;
  return symbol;
}

void declare(mlir::ModuleOp module, llvm::StringRef qualifiedName) {
  std::string symbol = symbolFor(qualifiedName);
  if (module.lookupSymbol(symbol))
    return;
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  mlir::LLVM::GlobalOp::create(builder, module.getLoc(),
                               declarationType(module.getContext()),
                               /*isConstant=*/true,
                               mlir::LLVM::Linkage::External, symbol,
                               mlir::Attribute());
}

mlir::Value classWord(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::ModuleOp module, llvm::StringRef qualifiedName) {
  declare(module, qualifiedName);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value address = mlir::LLVM::AddressOfOp::create(
      builder, loc, ptrType, symbolFor(qualifiedName));
  return mlir::LLVM::PtrToIntOp::create(builder, loc, builder.getI64Type(),
                                        address);
}

mlir::Value classWord(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef qualifiedName) {
  return classWord(builder, loc, enclosingModule(builder), qualifiedName);
}

namespace {

// A runtime class's bases, primary first: CPython's -- the exception
// taxonomy's for an exception (with ExceptionGroup's second base), int for
// bool, object otherwise. Empty for object.
std::vector<std::string> runtimeBases(llvm::StringRef contract) {
  if (contract == "builtins.object")
    return {};
  if (contract == "builtins.bool")
    return {"builtins.int"};
  if (const exceptions::BuiltinExceptionInfo *info =
          exceptions::findByContract(contract)) {
    std::vector<std::string> bases{info->baseContract.empty()
                                       ? std::string("builtins.object")
                                       : info->baseContract.str()};
    for (const exceptions::BuiltinExceptionExtraEdge &edge :
         exceptions::kBuiltinExceptionExtraEdges)
      if (edge.contract == contract)
        bases.push_back(edge.extraBaseContract.str());
    return bases;
  }
  return {"builtins.object"};
}

// C3, as CPython's mro_implementation: the class, then the merge of its
// bases' MROs and the bases themselves.
std::vector<std::string> runtimeMro(llvm::StringRef contract) {
  std::vector<std::vector<std::string>> sequences;
  std::vector<std::string> bases = runtimeBases(contract);
  for (const std::string &base : bases)
    sequences.push_back(runtimeMro(base));
  sequences.push_back(bases);
  std::vector<std::string> mro{contract.str()};
  for (;;) {
    llvm::erase_if(sequences, [](const auto &sequence) { return sequence.empty(); });
    if (sequences.empty())
      return mro;
    std::string head;
    for (const auto &sequence : sequences) {
      const std::string &candidate = sequence.front();
      bool inTail = llvm::any_of(sequences, [&](const auto &other) {
        return llvm::is_contained(llvm::drop_begin(other), candidate);
      });
      if (!inTail) {
        head = candidate;
        break;
      }
    }
    // The table is consistent; a cycle would be a table bug, and the order
    // so far is still a valid prefix.
    if (head.empty())
      return mro;
    mro.push_back(head);
    for (auto &sequence : sequences)
      if (!sequence.empty() && sequence.front() == head)
        sequence.erase(sequence.begin());
  }
}

} // namespace

std::optional<Definition> runtimeDefinition(llvm::StringRef contract) {
  if (!runtime_classes::isListed(contract))
    return std::nullopt;
  Definition definition;
  definition.qualifiedName = contract.str();
  definition.name = contracts::displayClassNameForContract(contract);
  // The one runtime class whose CPython `__module__` is not its manifest's:
  // io.UnsupportedOperation is defined by io, not _io.
  if (contract == "_io.UnsupportedOperation")
    definition.name = "io.UnsupportedOperation";
  if (contract == "builtins.object")
    return definition;
  definition.base = runtimeBases(contract).front();
  definition.mro = runtimeMro(contract);
  definition.mro.erase(definition.mro.begin());
  return definition;
}

void recordSource(mlir::ModuleOp module, const Definition &definition) {
  mlir::Builder builder(module.getContext());
  llvm::SmallVector<mlir::Attribute, 16> entries;
  if (auto recorded = module->getAttrOfType<mlir::ArrayAttr>(kSourceTypesAttr))
    entries.append(recorded.begin(), recorded.end());
  llvm::SmallVector<llvm::StringRef, 8> fields{
      definition.qualifiedName, definition.name, definition.base};
  for (const std::string &ancestor : definition.mro)
    fields.push_back(ancestor);
  entries.push_back(builder.getStrArrayAttr(fields));
  module->setAttr(kSourceTypesAttr, builder.getArrayAttr(entries));
}

mlir::LogicalResult defineDeclared(mlir::ModuleOp module,
                                   unsigned pointerBits) {
  // A definition declares its base, so walk until nothing new is declared.
  llvm::StringSet<> done;
  // A class with a slot is defined whether or not this module names it.
  if (auto table = module->getAttrOfType<mlir::DictionaryAttr>(kSlotsAttr))
    for (mlir::NamedAttribute entry : table)
      declare(module, entry.getName().getValue());
  for (bool changed = true; changed;) {
    changed = false;
    llvm::SmallVector<std::string, 16> pending;
    for (auto global : module.getOps<mlir::LLVM::GlobalOp>())
      if (declaresOnly(global))
        if (std::optional<llvm::StringRef> name =
                qualifiedNameOf(global.getSymName()))
          if (!done.contains(*name))
            pending.push_back(name->str());
    for (const std::string &name : pending) {
      done.insert(name);
      std::optional<Definition> definition = recordedSource(module, name);
      if (!definition)
        definition = runtimeDefinition(name);
      if (!definition)
        return module.emitError()
               << "class '" << name
               << "' is named by a class word but has no type object: it is "
                  "neither a runtime class nor a class this module declares";
      if (mlir::failed(defineOne(module, *definition, pointerBits)))
        return mlir::failure();
      changed = true;
    }
  }
  module->removeAttr(kSourceTypesAttr);
  module->removeAttr(kSlotsAttr);
  return mlir::success();
}

const SlotHook *slotHookNamed(llvm::StringRef hookName) {
  for (const SlotHook &hook : kSlotHooks)
    if (hook.name == hookName)
      return &hook;
  return nullptr;
}

void recordSlot(mlir::ModuleOp module, llvm::StringRef qualifiedName,
                const SlotHook &hook, llvm::StringRef slot) {
  mlir::Builder builder(module.getContext());
  mlir::NamedAttrList table;
  if (auto existing = module->getAttrOfType<mlir::DictionaryAttr>(kSlotsAttr))
    table.append(existing.getValue());
  mlir::NamedAttrList slots;
  if (auto existing =
          mlir::dyn_cast_if_present<mlir::DictionaryAttr>(table.get(qualifiedName)))
    slots.append(existing.getValue());
  slots.set(std::to_string(hook.word), builder.getStringAttr(slot));
  table.set(qualifiedName, slots.getDictionary(module.getContext()));
  module->setAttr(kSlotsAttr, table.getDictionary(module.getContext()));
}

mlir::LogicalResult defineSlotHooks(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();
  for (const SlotHook &hook : kSlotHooks) {
    auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(hook.name);
    if (!function && hook.calledByRuntime) {
      // Its only callers are in the runtime: `(ptr, i64) -> i1`.
      builder.setInsertionPointToEnd(module.getBody());
      function = mlir::LLVM::LLVMFuncOp::create(
          builder, module.getLoc(), hook.name,
          mlir::LLVM::LLVMFunctionType::get(i1, {ptrType, i64}));
    }
    if (!function || !function.isExternal())
      continue;
    mlir::SymbolTable::setSymbolVisibility(
        function, mlir::SymbolTable::Visibility::Public);
    mlir::LLVM::LLVMFunctionType type = function.getFunctionType();
    unsigned arguments = hook.binary ? 4 : 2;
    if (type.getNumParams() != arguments)
      return function.emitError()
             << "a boxed hook takes " << arguments << " arguments";
    mlir::Location loc = function.getLoc();
    mlir::Type resultType = type.getReturnType();
    function.setLinkage(mlir::LLVM::Linkage::External);
    mlir::Block *entry = function.addEntryBlock(builder);
    mlir::Region &body = function.getBody();
    mlir::Block *read = builder.createBlock(&body);
    mlir::Block *call = builder.createBlock(&body, body.end(), {ptrType}, {loc});
    mlir::Block *miss = builder.createBlock(&body);
    mlir::Value classWord = entry->getArgument(hook.binary ? 2 : 1);

    builder.setInsertionPointToEnd(entry);
    mlir::Value zero = mlir::LLVM::ConstantOp::create(
        builder, loc, i64, builder.getI64IntegerAttr(0));
    mlir::LLVM::CondBrOp::create(
        builder, loc,
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   classWord, zero),
        miss, read);

    builder.setInsertionPointToEnd(read);
    mlir::Value type0 =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrType, classWord);
    mlir::Value slotAt = mlir::LLVM::GEPOp::create(
        builder, loc, ptrType, i64, type0,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{
            static_cast<std::int32_t>(hook.word)});
    mlir::Value slot = mlir::LLVM::LoadOp::create(builder, loc, ptrType, slotAt,
                                                  /*alignment=*/8);
    mlir::Value none = mlir::LLVM::ZeroOp::create(builder, loc, ptrType);
    mlir::LLVM::CondBrOp::create(
        builder, loc,
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   slot, none),
        miss, mlir::ValueRange{}, call, mlir::ValueRange{slot});

    builder.setInsertionPointToEnd(call);
    llvm::SmallVector<mlir::Type, 3> slotParams{ptrType};
    llvm::SmallVector<mlir::Value, 4> operands{call->getArgument(0),
                                               entry->getArgument(0)};
    if (hook.binary) {
      slotParams.append({ptrType, i64});
      operands.append({entry->getArgument(1), entry->getArgument(3)});
    }
    auto slotType = mlir::LLVM::LLVMFunctionType::get(resultType, slotParams);
    auto result = mlir::LLVM::CallOp::create(builder, loc, slotType, operands);
    mlir::LLVM::ReturnOp::create(builder, loc, result.getResults());

    builder.setInsertionPointToEnd(miss);
    mlir::Value missed = mlir::LLVM::ConstantOp::create(
        builder, loc, i1, builder.getBoolAttr(false));
    if (auto results = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(resultType)) {
      mlir::Value poison =
          mlir::LLVM::PoisonOp::create(builder, loc, resultType);
      missed = mlir::LLVM::InsertValueOp::create(
          builder, loc, poison, missed,
          llvm::ArrayRef<std::int64_t>{
              static_cast<std::int64_t>(results.getBody().size()) - 1});
    } else if (resultType != i1) {
      return function.emitError() << "a boxed hook answers with a hit bit";
    }
    mlir::LLVM::ReturnOp::create(builder, loc, missed);
  }
  return mlir::success();
}

mlir::LogicalResult resolveManifestClassWords(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::arith::ConstantOp, 64> constants;
  module.walk([&](mlir::arith::ConstantOp constant) {
    if (constant->hasAttr(kClassWordOfAttr))
      constants.push_back(constant);
  });
  for (mlir::arith::ConstantOp constant : constants) {
    auto name = constant->getAttrOfType<mlir::StringAttr>(kClassWordOfAttr);
    if (!name || !runtime_classes::isListed(name.getValue()) ||
        !constant.getType().isInteger(64))
      return constant.emitError()
             << "ly.class_of must name a runtime class (RuntimeClasses.h) "
                "on an i64 constant";
    mlir::OpBuilder builder(constant);
    mlir::Value word =
        classWord(builder, constant.getLoc(), module, name.getValue());
    constant.getResult().replaceAllUsesWith(word);
    constant.erase();
  }
  return mlir::success();
}

mlir::FailureOr<StaticClassWords>
collectStaticClassWords(mlir::ModuleOp module) {
  StaticClassWords words;
  for (auto global : module.getOps<mlir::memref::GlobalOp>()) {
    auto name = global->getAttrOfType<mlir::StringAttr>(kClassWordOfAttr);
    if (!name)
      continue;
    if (!runtime_classes::isListed(name.getValue()))
      return global.emitError() << "ly.class_of names '" << name.getValue()
                                << "', which is not a runtime class";
    StaticClassWords::Entry entry;
    entry.qualifiedName = name.getValue().str();
    entry.shared = global->hasAttr(kSharedStaticAttr);
    if (auto stride =
            global->getAttrOfType<mlir::IntegerAttr>(kClassWordStrideAttr))
      entry.stride = stride.getInt();
    words.globals[global.getSymName()] = std::move(entry);
  }
  return words;
}

mlir::LogicalResult patchStaticClassWords(mlir::ModuleOp module,
                                          const StaticClassWords &words,
                                          unsigned pointerBits) {
  mlir::OpBuilder builder(module.getContext());
  for (const auto &item : words.globals) {
    auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(item.getKey());
    // Erased as unused since it was recorded.
    if (!global)
      continue;
    auto initial =
        mlir::dyn_cast_if_present<mlir::DenseIntElementsAttr>(global.getValueOrNull());
    if (!initial)
      return global.emitError()
             << "a static object with a class word needs a dense integer "
                "initializer";
    unsigned width = initial.getElementType().getIntOrFloatBitWidth();
    llvm::SmallVector<std::int8_t, 64> bytes;
    for (const llvm::APInt &value : initial.getValues<llvm::APInt>())
      for (unsigned byte = 0; byte < width / 8; ++byte)
        bytes.push_back(static_cast<std::int8_t>(
            value.extractBitsAsZExtValue(8, 8 * byte)));
    std::int64_t stride = item.getValue().stride
                              ? item.getValue().stride
                              : static_cast<std::int64_t>(bytes.size());
    // A single record (a str's image) may end in a partial word; records
    // that repeat must each start on one.
    bool single = stride == static_cast<std::int64_t>(bytes.size());
    if (stride < 16 || (!single && stride % 8 != 0) ||
        static_cast<std::int64_t>(bytes.size()) % stride != 0)
      return global.emitError()
             << "a static object's class word needs records of at least two "
                "words";
    std::string symbol = symbolFor(item.getValue().qualifiedName);
    declare(module, item.getValue().qualifiedName);
    llvm::SmallVector<ImageField, 64> fields;
    for (std::int64_t record = 0;
         record < static_cast<std::int64_t>(bytes.size()); record += stride) {
      auto appendBytes = [&](std::int64_t from, std::int64_t to) {
        appendImageBytes(fields, llvm::ArrayRef<std::int8_t>(bytes).slice(
                                     from, to - from));
      };
      appendBytes(record, record + 8);
      appendImageAddress(fields, symbol);
      appendBytes(record + 16, record + stride);
    }
    builder.setInsertionPoint(global);
    setImage(builder, global.getLoc(), global, fields, pointerBits);
    // ⛔ A packed struct is aligned to 1, and an object's address must keep
    // its two low bits clear: they are what tells it from an immediate.
    if (global.getAlignment().value_or(0) < 8)
      global.setAlignment(8);
    // One object for the whole program, whichever modules carry it.
    if (item.getValue().shared)
      global.setLinkage(mlir::LLVM::Linkage::LinkonceODR);
  }
  return mlir::success();
}

unsigned pointerBitsOf(mlir::ModuleOp module) {
  if (std::optional<native::TargetPlatformFacts> facts =
          native::readTargetPlatformFacts(module))
    return static_cast<unsigned>(facts->pointerWidth);
  if (auto layout = module->getAttrOfType<mlir::StringAttr>(
          mlir::LLVM::LLVMDialect::getDataLayoutAttrName()))
    return llvm::DataLayout(layout.getValue()).getPointerSizeInBits();
  if (auto triple = module->getAttrOfType<mlir::StringAttr>(
          native::kTargetTripleAttr))
    return llvm::Triple(triple.getValue()).isArch32Bit() ? 32 : 64;
  return 64;
}

} // namespace py::type_objects
