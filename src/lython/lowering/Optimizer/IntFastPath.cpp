#include "Optimizer/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"

#include <memory>
#include <optional>

namespace py::optimizer {
namespace {

constexpr llvm::StringLiteral kFastPathAttr{"ly.int_fast_path"};
constexpr llvm::StringLiteral kFastPathTargetAttr{"ly.int_fast_path_target"};

bool parseI64(llvm::StringRef text, int64_t &result) {
  llvm::SmallString<32> cleaned;
  cleaned.reserve(text.size());
  for (char ch : text) {
    if (ch != '_')
      cleaned.push_back(ch);
  }
  return !llvm::StringRef(cleaned).getAsInteger(10, result);
}

bool fitsSigned(int64_t value, unsigned bits) {
  if (bits >= 64)
    return true;
  int64_t min = -(int64_t{1} << (bits - 1));
  int64_t max = (int64_t{1} << (bits - 1)) - 1;
  return value >= min && value <= max;
}

bool checkedAdd(int64_t lhs, int64_t rhs, unsigned bits, int64_t &result) {
  if (!fitsSigned(lhs, bits) || !fitsSigned(rhs, bits))
    return false;
  bool overflowed = false;
  llvm::APInt value =
      llvm::APInt(bits, static_cast<uint64_t>(lhs), /*isSigned=*/true)
          .sadd_ov(llvm::APInt(bits, static_cast<uint64_t>(rhs),
                               /*isSigned=*/true),
                   overflowed);
  if (overflowed)
    return false;
  result = value.getSExtValue();
  return true;
}

bool checkedSub(int64_t lhs, int64_t rhs, unsigned bits, int64_t &result) {
  if (!fitsSigned(lhs, bits) || !fitsSigned(rhs, bits))
    return false;
  bool overflowed = false;
  llvm::APInt value =
      llvm::APInt(bits, static_cast<uint64_t>(lhs), /*isSigned=*/true)
          .ssub_ov(llvm::APInt(bits, static_cast<uint64_t>(rhs),
                               /*isSigned=*/true),
                   overflowed);
  if (overflowed)
    return false;
  result = value.getSExtValue();
  return true;
}

bool checkedMul(int64_t lhs, int64_t rhs, unsigned bits, int64_t &result) {
  if (!fitsSigned(lhs, bits) || !fitsSigned(rhs, bits))
    return false;
  bool overflowed = false;
  llvm::APInt value =
      llvm::APInt(bits, static_cast<uint64_t>(lhs), /*isSigned=*/true)
          .smul_ov(llvm::APInt(bits, static_cast<uint64_t>(rhs),
                               /*isSigned=*/true),
                   overflowed);
  if (overflowed)
    return false;
  result = value.getSExtValue();
  return true;
}

std::optional<bool> evalCompare(mlir::Operation *op, int64_t lhs, int64_t rhs) {
  if (mlir::isa<LeOp>(op))
    return lhs <= rhs;
  if (mlir::isa<LtOp>(op))
    return lhs < rhs;
  if (mlir::isa<GtOp>(op))
    return lhs > rhs;
  if (mlir::isa<GeOp>(op))
    return lhs >= rhs;
  if (mlir::isa<EqOp>(op))
    return lhs == rhs;
  if (mlir::isa<NeOp>(op))
    return lhs != rhs;
  return std::nullopt;
}

mlir::arith::CmpIPredicate predicateFor(mlir::Operation *op) {
  if (mlir::isa<LeOp>(op))
    return mlir::arith::CmpIPredicate::sle;
  if (mlir::isa<LtOp>(op))
    return mlir::arith::CmpIPredicate::slt;
  if (mlir::isa<GtOp>(op))
    return mlir::arith::CmpIPredicate::sgt;
  if (mlir::isa<GeOp>(op))
    return mlir::arith::CmpIPredicate::sge;
  if (mlir::isa<EqOp>(op))
    return mlir::arith::CmpIPredicate::eq;
  return mlir::arith::CmpIPredicate::ne;
}

FuncOp directTarget(mlir::Operation *from, mlir::Value callable) {
  callable = value::stripCasts(callable);
  auto funcObject = callable.getDefiningOp<FuncObjectOp>();
  if (!funcObject)
    return nullptr;
  mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
      from, funcObject.getTargetAttr());
  return mlir::dyn_cast_or_null<FuncOp>(symbol);
}

bool isEmptyTuple(mlir::Value value) {
  return mlir::isa_and_nonnull<TupleEmptyOp>(
      value::stripCasts(value).getDefiningOp());
}

std::optional<mlir::Value> singlePosArg(CallVectorOp call) {
  if (!isEmptyTuple(call.getKwnames()) || !isEmptyTuple(call.getKwvalues()))
    return std::nullopt;
  auto tuple =
      value::stripCasts(call.getPosargs()).getDefiningOp<TupleCreateOp>();
  if (!tuple || tuple.getElements().size() != 1)
    return std::nullopt;
  return tuple.getElements().front();
}

bool isIntUnarySignature(FuncOp func) {
  auto attr = func->getAttrOfType<mlir::TypeAttr>("function_type");
  auto sig = attr ? mlir::dyn_cast<FuncSignatureType>(attr.getValue())
                  : FuncSignatureType();
  if (!sig || sig.hasVararg() || sig.hasKwarg())
    return false;
  return sig.getPositionalTypes().size() == 1 &&
         sig.getResultTypes().size() == 1 &&
         mlir::isa<IntType>(sig.getPositionalTypes().front()) &&
         mlir::isa<IntType>(sig.getResultTypes().front());
}

struct EvalFrame {
  llvm::DenseMap<mlir::Value, int64_t> ints;
  llvm::DenseMap<mlir::Value, bool> bools;

  std::optional<int64_t> intValue(mlir::Value value) const {
    auto it = ints.find(value);
    if (it == ints.end())
      return std::nullopt;
    return it->second;
  }

  std::optional<bool> boolValue(mlir::Value value) const {
    auto it = bools.find(value);
    if (it == bools.end())
      return std::nullopt;
    return it->second;
  }
};

class IntEvaluator {
public:
  IntEvaluator(FuncOp func, unsigned bits) : func(func), bits(bits) {}

  std::optional<int64_t> evaluate(int64_t input) {
    if (!fitsSigned(input, bits))
      return std::nullopt;
    if (auto it = memo.find(input); it != memo.end())
      return it->second;
    if (!active.insert(input).second)
      return std::nullopt;
    std::optional<int64_t> result = evaluateUncached(input);
    active.erase(input);
    if (result)
      memo[input] = *result;
    return result;
  }

private:
  std::optional<int64_t> evaluateUncached(int64_t input) {
    if (!func || func.getBody().empty() || !isIntUnarySignature(func))
      return std::nullopt;

    mlir::Block *block = &func.getBody().front();
    if (block->getNumArguments() != 1)
      return std::nullopt;

    EvalFrame frame;
    frame.ints[block->getArgument(0)] = input;
    for (unsigned step = 0; step < 10000; ++step) {
      bool branched = false;
      for (mlir::Operation &operation : *block) {
        mlir::Operation *op = &operation;
        if (auto constant = mlir::dyn_cast<IntConstantOp>(op)) {
          int64_t value = 0;
          if (!parseI64(constant.getValue(), value))
            return std::nullopt;
          frame.ints[constant.getResult()] = value;
          continue;
        }
        if (mlir::isa<LeOp, LtOp, GtOp, GeOp, EqOp, NeOp>(op)) {
          auto lhs = frame.intValue(op->getOperand(0));
          auto rhs = frame.intValue(op->getOperand(1));
          if (!lhs || !rhs)
            return std::nullopt;
          std::optional<bool> result = evalCompare(op, *lhs, *rhs);
          if (!result)
            return std::nullopt;
          frame.bools[op->getResult(0)] = *result;
          continue;
        }
        if (auto cast = mlir::dyn_cast<CastToPrimOp>(op)) {
          auto bit = frame.boolValue(cast.getInput());
          if (!bit || !cast.getResult().getType().isInteger(1))
            return std::nullopt;
          frame.bools[cast.getResult()] = *bit;
          continue;
        }
        if (mlir::isa<AddOp, SubOp, MulOp>(op)) {
          auto lhs = frame.intValue(op->getOperand(0));
          auto rhs = frame.intValue(op->getOperand(1));
          if (!lhs || !rhs)
            return std::nullopt;
          int64_t result = 0;
          bool ok = mlir::isa<AddOp>(op) ? checkedAdd(*lhs, *rhs, bits, result)
                    : mlir::isa<SubOp>(op)
                        ? checkedSub(*lhs, *rhs, bits, result)
                        : checkedMul(*lhs, *rhs, bits, result);
          if (!ok)
            return std::nullopt;
          frame.ints[op->getResult(0)] = result;
          continue;
        }
        if (auto call = mlir::dyn_cast<CallVectorOp>(op)) {
          if (directTarget(call, call.getCallable()) != func)
            return std::nullopt;
          std::optional<mlir::Value> arg = singlePosArg(call);
          if (!arg)
            return std::nullopt;
          auto value = frame.intValue(*arg);
          if (!value)
            return std::nullopt;
          auto result = evaluate(*value);
          if (!result)
            return std::nullopt;
          frame.ints[call.getResult(0)] = *result;
          continue;
        }
        if (mlir::isa<FuncObjectOp, TupleCreateOp, TupleEmptyOp>(op))
          continue;
        if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(op)) {
          auto condition = frame.boolValue(cond.getCondition());
          if (!condition)
            return std::nullopt;
          auto operands =
              *condition ? cond.getTrueOperands() : cond.getFalseOperands();
          block = *condition ? cond.getTrueDest() : cond.getFalseDest();
          if (!mapBlockArgs(frame, block, operands))
            return std::nullopt;
          branched = true;
          break;
        }
        if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(op)) {
          block = branch.getDest();
          if (!mapBlockArgs(frame, block, branch.getDestOperands()))
            return std::nullopt;
          branched = true;
          break;
        }
        if (auto ret = mlir::dyn_cast<ReturnOp>(op)) {
          if (ret->getNumOperands() != 1)
            return std::nullopt;
          return frame.intValue(ret->getOperand(0));
        }
        return std::nullopt;
      }
      if (!branched)
        return std::nullopt;
    }
    return std::nullopt;
  }

  static bool mapBlockArgs(EvalFrame &frame, mlir::Block *block,
                           mlir::ValueRange operands) {
    if (!block || operands.size() != block->getNumArguments())
      return false;
    for (auto [arg, operand] : llvm::zip(block->getArguments(), operands)) {
      if (auto value = frame.intValue(operand)) {
        frame.ints[arg] = *value;
        continue;
      }
      if (auto bit = frame.boolValue(operand)) {
        frame.bools[arg] = *bit;
        continue;
      }
      return false;
    }
    return true;
  }

  FuncOp func;
  unsigned bits;
  llvm::DenseMap<int64_t, int64_t> memo;
  llvm::DenseSet<int64_t> active;
};

class HelperBuilder {
public:
  HelperBuilder(FuncOp source, mlir::func::FuncOp helper, unsigned bits)
      : source(source), helper(helper), bits(bits) {}

  bool build() {
    if (!source || !helper || source.getBody().empty())
      return false;
    createBlocks();
    for (mlir::Block &block : source.getBody())
      if (!cloneBlock(block))
        return false;
    return true;
  }

private:
  void createBlocks() {
    mlir::Region &src = source.getBody();
    mlir::Region &dst = helper.getBody();
    mlir::Block *entry = helper.addEntryBlock();
    blockMap[&src.front()] = entry;
    mapArgs(src.front(), *entry);

    for (mlir::Block &block : llvm::drop_begin(src, 1)) {
      auto *copy = new mlir::Block();
      for (mlir::BlockArgument arg : block.getArguments())
        copy->addArgument(loweredType(arg.getType()), arg.getLoc());
      dst.push_back(copy);
      blockMap[&block] = copy;
      mapArgs(block, *copy);
    }
  }

  mlir::Type loweredType(mlir::Type type) const {
    if (mlir::isa<IntType>(type))
      return mlir::IntegerType::get(type.getContext(), bits);
    if (mlir::isa<BoolType>(type) || type.isInteger(1))
      return mlir::IntegerType::get(type.getContext(), 1);
    if (type.isInteger(64))
      return type;
    return type;
  }

  void mapArgs(mlir::Block &from, mlir::Block &to) {
    for (auto [srcArg, dstArg] :
         llvm::zip(from.getArguments(), to.getArguments()))
      valueMap[srcArg] = dstArg;
  }

  std::optional<mlir::Value> mapped(mlir::Value value) const {
    auto it = valueMap.find(value);
    if (it == valueMap.end())
      return std::nullopt;
    return it->second;
  }

  std::optional<llvm::SmallVector<mlir::Value, 4>>
  mappedTuple(mlir::Value value) const {
    auto it = tupleMap.find(value::stripCasts(value));
    if (it == tupleMap.end())
      return std::nullopt;
    return it->second;
  }

  bool mapOperands(mlir::ValueRange source,
                   llvm::SmallVectorImpl<mlir::Value> &dest) const {
    for (mlir::Value operand : source) {
      auto value = mapped(operand);
      if (!value)
        return false;
      dest.push_back(*value);
    }
    return true;
  }

  bool cloneBlock(mlir::Block &sourceBlock) {
    mlir::Block *destBlock = blockMap.lookup(&sourceBlock);
    if (!destBlock)
      return false;

    mlir::OpBuilder builder(destBlock, destBlock->end());
    for (mlir::Operation &operation : sourceBlock) {
      mlir::Operation *op = &operation;
      mlir::Location loc = op->getLoc();
      if (auto constant = mlir::dyn_cast<IntConstantOp>(op)) {
        int64_t value = 0;
        if (!parseI64(constant.getValue(), value))
          return false;
        valueMap[constant.getResult()] =
            builder.create<mlir::arith::ConstantIntOp>(loc, value, bits);
        continue;
      }
      if (mlir::isa<LeOp, LtOp, GtOp, GeOp, EqOp, NeOp>(op)) {
        auto lhs = mapped(op->getOperand(0));
        auto rhs = mapped(op->getOperand(1));
        if (!lhs || !rhs)
          return false;
        valueMap[op->getResult(0)] = builder.create<mlir::arith::CmpIOp>(
            loc, predicateFor(op), *lhs, *rhs);
        continue;
      }
      if (auto cast = mlir::dyn_cast<CastToPrimOp>(op)) {
        auto input = mapped(cast.getInput());
        if (!input)
          return false;
        valueMap[cast.getResult()] = *input;
        continue;
      }
      if (mlir::isa<AddOp, SubOp, MulOp>(op)) {
        auto lhs = mapped(op->getOperand(0));
        auto rhs = mapped(op->getOperand(1));
        if (!lhs || !rhs)
          return false;
        mlir::Value result =
            mlir::isa<AddOp>(op)
                ? builder.create<mlir::arith::AddIOp>(loc, *lhs, *rhs)
                      .getResult()
            : mlir::isa<SubOp>(op)
                ? builder.create<mlir::arith::SubIOp>(loc, *lhs, *rhs)
                      .getResult()
                : builder.create<mlir::arith::MulIOp>(loc, *lhs, *rhs)
                      .getResult();
        valueMap[op->getResult(0)] = result;
        continue;
      }
      if (auto tuple = mlir::dyn_cast<TupleCreateOp>(op)) {
        llvm::SmallVector<mlir::Value, 4> elements;
        if (!mapOperands(tuple.getElements(), elements))
          return false;
        tupleMap[tuple.getResult()] = elements;
        continue;
      }
      if (auto tuple = mlir::dyn_cast<TupleEmptyOp>(op)) {
        tupleMap[tuple.getResult()] = {};
        continue;
      }
      if (mlir::isa<FuncObjectOp>(op))
        continue;
      if (auto call = mlir::dyn_cast<CallVectorOp>(op)) {
        if (directTarget(call, call.getCallable()) != source)
          return false;
        auto args = mappedTuple(call.getPosargs());
        if (!args || args->size() != 1 || !isEmptyTuple(call.getKwnames()) ||
            !isEmptyTuple(call.getKwvalues()))
          return false;
        auto result = builder.create<mlir::func::CallOp>(
            loc, helper.getSymName(),
            mlir::TypeRange{builder.getIntegerType(bits)}, *args);
        valueMap[call.getResult(0)] = result.getResult(0);
        continue;
      }
      if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(op)) {
        auto condition = mapped(cond.getCondition());
        if (!condition)
          return false;
        llvm::SmallVector<mlir::Value> trueOperands;
        llvm::SmallVector<mlir::Value> falseOperands;
        if (!mapOperands(cond.getTrueOperands(), trueOperands) ||
            !mapOperands(cond.getFalseOperands(), falseOperands))
          return false;
        builder.create<mlir::cf::CondBranchOp>(
            loc, *condition, blockMap.lookup(cond.getTrueDest()), trueOperands,
            blockMap.lookup(cond.getFalseDest()), falseOperands);
        continue;
      }
      if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(op)) {
        llvm::SmallVector<mlir::Value> operands;
        if (!mapOperands(branch.getDestOperands(), operands))
          return false;
        builder.create<mlir::cf::BranchOp>(
            loc, blockMap.lookup(branch.getDest()), operands);
        continue;
      }
      if (auto ret = mlir::dyn_cast<ReturnOp>(op)) {
        llvm::SmallVector<mlir::Value> operands;
        if (!mapOperands(ret.getOperands(), operands))
          return false;
        builder.create<mlir::func::ReturnOp>(loc, operands);
        continue;
      }
      return false;
    }
    return true;
  }

  FuncOp source;
  mlir::func::FuncOp helper;
  unsigned bits;
  llvm::DenseMap<mlir::Block *, mlir::Block *> blockMap;
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>> tupleMap;
};

std::string helperName(FuncOp func, unsigned bits) {
  return (func.getSymName() + "$int_i" + llvm::Twine(bits)).str();
}

mlir::func::FuncOp getOrCreateHelper(FuncOp func, mlir::ModuleOp module,
                                     unsigned bits) {
  std::string name = helperName(func, bits);
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;

  mlir::OpBuilder builder(module.getBodyRegion());
  auto intType = builder.getIntegerType(bits);
  auto helper = builder.create<mlir::func::FuncOp>(
      func.getLoc(), name, builder.getFunctionType({intType}, {intType}));
  helper.setPrivate();
  helper->setAttr(kFastPathAttr, builder.getUnitAttr());
  helper->setAttr(
      kFastPathTargetAttr,
      mlir::FlatSymbolRefAttr::get(func.getContext(), func.getSymName()));
  if (!HelperBuilder(func, helper, bits).build()) {
    helper.erase();
    return nullptr;
  }
  return helper;
}

void eraseDeadStaticCallTree(mlir::Value value) {
  value = value::stripCasts(value);
  if (!value)
    return;
  if (auto tuple = value.getDefiningOp<TupleCreateOp>()) {
    if (!tuple->use_empty())
      return;
    llvm::SmallVector<mlir::Value> elements(tuple.getElements().begin(),
                                            tuple.getElements().end());
    tuple.erase();
    for (mlir::Value element : elements)
      eraseDeadStaticCallTree(element);
    return;
  }
  if (auto tuple = value.getDefiningOp<TupleEmptyOp>()) {
    if (tuple->use_empty())
      tuple.erase();
    return;
  }
  if (auto func = value.getDefiningOp<FuncObjectOp>()) {
    if (func->use_empty())
      func.erase();
    return;
  }
  if (auto constant = value.getDefiningOp<IntConstantOp>())
    if (constant->use_empty())
      constant.erase();
}

void rewriteCall(CallVectorOp call, mlir::func::FuncOp helper) {
  std::optional<mlir::Value> arg = singlePosArg(call);
  if (!arg)
    return;

  mlir::OpBuilder builder(call);
  auto functionType = helper.getFunctionType();
  mlir::Type intType = functionType.getInput(0);
  auto unboxed = builder.create<CastToPrimOp>(call.getLoc(), intType, *arg,
                                              builder.getStringAttr("exact"));
  auto fast = builder.create<mlir::func::CallOp>(
      call.getLoc(), helper.getSymName(), mlir::TypeRange{intType},
      mlir::ValueRange{unboxed.getResult()});
  auto boxed = builder.create<CastFromPrimOp>(
      call.getLoc(), call.getResult(0).getType(), fast.getResult(0));

  mlir::Value oldCallable = call.getCallable();
  mlir::Value oldPosargs = call.getPosargs();
  mlir::Value oldKwnames = call.getKwnames();
  mlir::Value oldKwvalues = call.getKwvalues();
  call.getResult(0).replaceAllUsesWith(boxed.getResult());
  call.erase();
  eraseDeadStaticCallTree(oldCallable);
  eraseDeadStaticCallTree(oldPosargs);
  eraseDeadStaticCallTree(oldKwnames);
  eraseDeadStaticCallTree(oldKwvalues);
}

} // namespace

void int_fastpath::specialize(mlir::ModuleOp module) {
  llvm::SmallVector<CallVectorOp> calls;
  module.walk([&](CallVectorOp call) { calls.push_back(call); });

  llvm::DenseMap<FuncOp, std::unique_ptr<IntEvaluator>> evaluators32;
  llvm::DenseMap<FuncOp, std::unique_ptr<IntEvaluator>> evaluators64;
  for (CallVectorOp call : calls) {
    FuncOp func = directTarget(call, call.getCallable());
    if (!func || !isIntUnarySignature(func))
      continue;
    std::optional<mlir::Value> arg = singlePosArg(call);
    if (!arg)
      continue;
    auto constant = value::stripCasts(*arg).getDefiningOp<IntConstantOp>();
    if (!constant)
      continue;
    int64_t input = 0;
    if (!parseI64(constant.getValue(), input))
      continue;

    unsigned bits = 0;
    auto &evaluator32 = evaluators32[func];
    if (!evaluator32)
      evaluator32 = std::make_unique<IntEvaluator>(func, 32);
    if (evaluator32->evaluate(input)) {
      bits = 32;
    } else {
      auto &evaluator64 = evaluators64[func];
      if (!evaluator64)
        evaluator64 = std::make_unique<IntEvaluator>(func, 64);
      if (!evaluator64->evaluate(input))
        continue;
      bits = 64;
    }

    mlir::func::FuncOp helper = getOrCreateHelper(func, module, bits);
    if (!helper)
      continue;
    rewriteCall(call, helper);
  }
}

} // namespace py::optimizer
