#include "DriverCodeGen.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "PyDialect.h.inc"

using namespace mlir;

namespace lython::driver {

static constexpr llvm::StringLiteral kLythonSafetyMetadataName{"ly.safety"};
static constexpr llvm::StringLiteral kLythonSafetyMetadataVersion{
    "ly.safety.v1"};
static constexpr llvm::StringLiteral kPySafetyContractIdAttr{
    "py.safety_contract_id"};

static std::optional<LLVMSafetyEffectKind>
getStructuralSafetyEffectKind(llvm::Instruction &inst) {
  if (llvm::isa<llvm::AtomicRMWInst>(inst))
    return LLVMSafetyEffectKind::AtomicRMW;
  if (llvm::isa<llvm::AtomicCmpXchgInst>(inst))
    return LLVMSafetyEffectKind::AtomicCmpXchg;
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
    if (load->isAtomic())
      return LLVMSafetyEffectKind::AtomicLoad;
  if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
    if (store->isAtomic())
      return LLVMSafetyEffectKind::AtomicStore;
  return std::nullopt;
}

static std::optional<llvm::AtomicRMWInst::BinOp>
mapAtomicBinOp(LLVM::AtomicBinOp op) {
  switch (op) {
  case LLVM::AtomicBinOp::xchg:
    return llvm::AtomicRMWInst::Xchg;
  case LLVM::AtomicBinOp::add:
    return llvm::AtomicRMWInst::Add;
  case LLVM::AtomicBinOp::sub:
    return llvm::AtomicRMWInst::Sub;
  case LLVM::AtomicBinOp::_and:
    return llvm::AtomicRMWInst::And;
  case LLVM::AtomicBinOp::nand:
    return llvm::AtomicRMWInst::Nand;
  case LLVM::AtomicBinOp::_or:
    return llvm::AtomicRMWInst::Or;
  case LLVM::AtomicBinOp::_xor:
    return llvm::AtomicRMWInst::Xor;
  case LLVM::AtomicBinOp::max:
    return llvm::AtomicRMWInst::Max;
  case LLVM::AtomicBinOp::min:
    return llvm::AtomicRMWInst::Min;
  case LLVM::AtomicBinOp::umax:
    return llvm::AtomicRMWInst::UMax;
  case LLVM::AtomicBinOp::umin:
    return llvm::AtomicRMWInst::UMin;
  case LLVM::AtomicBinOp::fadd:
    return llvm::AtomicRMWInst::FAdd;
  case LLVM::AtomicBinOp::fsub:
    return llvm::AtomicRMWInst::FSub;
  case LLVM::AtomicBinOp::fmax:
    return llvm::AtomicRMWInst::FMax;
  case LLVM::AtomicBinOp::fmin:
    return llvm::AtomicRMWInst::FMin;
  case LLVM::AtomicBinOp::fmaximum:
    return llvm::AtomicRMWInst::FMaximum;
  case LLVM::AtomicBinOp::fminimum:
    return llvm::AtomicRMWInst::FMinimum;
  case LLVM::AtomicBinOp::uinc_wrap:
    return llvm::AtomicRMWInst::UIncWrap;
  case LLVM::AtomicBinOp::udec_wrap:
    return llvm::AtomicRMWInst::UDecWrap;
  case LLVM::AtomicBinOp::usub_cond:
    return llvm::AtomicRMWInst::USubCond;
  case LLVM::AtomicBinOp::usub_sat:
    return llvm::AtomicRMWInst::USubSat;
  }
  return std::nullopt;
}

static std::optional<llvm::AtomicOrdering>
mapAtomicOrdering(LLVM::AtomicOrdering ordering) {
  switch (ordering) {
  case LLVM::AtomicOrdering::not_atomic:
    return llvm::AtomicOrdering::NotAtomic;
  case LLVM::AtomicOrdering::unordered:
    return llvm::AtomicOrdering::Unordered;
  case LLVM::AtomicOrdering::monotonic:
    return llvm::AtomicOrdering::Monotonic;
  case LLVM::AtomicOrdering::acquire:
    return llvm::AtomicOrdering::Acquire;
  case LLVM::AtomicOrdering::release:
    return llvm::AtomicOrdering::Release;
  case LLVM::AtomicOrdering::acq_rel:
    return llvm::AtomicOrdering::AcquireRelease;
  case LLVM::AtomicOrdering::seq_cst:
    return llvm::AtomicOrdering::SequentiallyConsistent;
  }
  return std::nullopt;
}

static std::optional<int64_t> mlirIntegerConstant(Value value) {
  if (auto constant = value.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
      return attr.getValue().getSExtValue();
  return std::nullopt;
}

static std::optional<int64_t> llvmIntegerConstant(llvm::Value *value) {
  auto *constant = llvm::dyn_cast_or_null<llvm::ConstantInt>(value);
  if (!constant)
    return std::nullopt;
  return constant->getSExtValue();
}

static bool sameAtomicRMWBinOp(llvm::AtomicRMWInst::BinOp actual,
                               const LLVMSafetyContract &contract) {
  if (!contract.rmwBinOp)
    return true;
  if (actual == *contract.rmwBinOp)
    return true;

  // LLVM's O2 pipeline canonicalizes no-op integer atomic reads in some
  // inlined helpers from `atomicrmw add 0` to `atomicrmw or 0`. The MLIR
  // thread-safety verifier has already validated the source contract, so keep
  // the post-optimization metadata check semantic for this one no-op shape.
  if (contract.integerOperand && *contract.integerOperand == 0 &&
      *contract.rmwBinOp == llvm::AtomicRMWInst::Add &&
      actual == llvm::AtomicRMWInst::Or)
    return true;

  return false;
}

static bool instructionMatchesContract(llvm::Instruction &inst,
                                       const LLVMSafetyContract &contract) {
  switch (contract.kind) {
  case LLVMSafetyEffectKind::AtomicRMW: {
    auto *rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&inst);
    if (!rmw)
      return false;
    if (!sameAtomicRMWBinOp(rmw->getOperation(), contract))
      return false;
    if (contract.integerOperand) {
      std::optional<int64_t> actual = llvmIntegerConstant(rmw->getValOperand());
      if (!actual || *actual != *contract.integerOperand)
        return false;
    }
    if (contract.ordering && rmw->getOrdering() != *contract.ordering)
      return false;
    return true;
  }
  case LLVMSafetyEffectKind::AtomicCmpXchg:
    return llvm::isa<llvm::AtomicCmpXchgInst>(inst);
  case LLVMSafetyEffectKind::AtomicLoad: {
    auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst);
    return load && load->isAtomic() &&
           (!contract.ordering || load->getOrdering() == *contract.ordering);
  }
  case LLVMSafetyEffectKind::AtomicStore: {
    auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst);
    return store && store->isAtomic() &&
           (!contract.ordering || store->getOrdering() == *contract.ordering);
  }
  }
  return false;
}

static std::optional<int64_t>
recoverDroppedAtomicRMWContract(llvm::Instruction &inst,
                                const LLVMSafetyProfile &profile,
                                llvm::ArrayRef<unsigned> reserved) {
  if (!llvm::isa<llvm::AtomicRMWInst>(&inst))
    return std::nullopt;

  llvm::Function *function = inst.getFunction();
  if (!function)
    return std::nullopt;
  std::optional<int64_t> fallback;
  for (const LLVMSafetyContract &contract : profile.contracts) {
    if (contract.functionName != function->getName())
      continue;
    if (contract.kind != LLVMSafetyEffectKind::AtomicRMW)
      continue;
    if (!instructionMatchesContract(inst, contract))
      continue;
    if (contract.id >= 0 &&
        static_cast<size_t>(contract.id) < reserved.size() &&
        reserved[static_cast<size_t>(contract.id)] == 0)
      return contract.id;
    if (!fallback)
      fallback = contract.id;
  }
  return fallback;
}

static std::optional<int64_t>
recoverDroppedSafetyContract(llvm::Instruction &inst,
                             const LLVMSafetyProfile &profile,
                             llvm::ArrayRef<unsigned> reserved) {
  return recoverDroppedAtomicRMWContract(inst, profile, reserved);
}

void collectLLVMSafetyContracts(ModuleOp module, LLVMSafetyProfile &profile) {
  module.walk([&](LLVM::LLVMFuncOp func) {
    for (Block &block : func.getBody()) {
      for (Operation &op : block) {
        LLVMSafetyContract contract;
        contract.id = static_cast<int64_t>(profile.contracts.size());
        contract.functionName = func.getName().str();
        auto markContract = [&] {
          op.setAttr(kPySafetyContractIdAttr,
                     IntegerAttr::get(IntegerType::get(op.getContext(), 64),
                                      contract.id));
          profile.contracts.push_back(std::move(contract));
        };

        if (auto atomic = dyn_cast<LLVM::AtomicRMWOp>(&op)) {
          contract.kind = LLVMSafetyEffectKind::AtomicRMW;
          contract.rmwBinOp = mapAtomicBinOp(atomic.getBinOp());
          contract.integerOperand = mlirIntegerConstant(atomic.getVal());
          contract.ordering = mapAtomicOrdering(atomic.getOrdering());
          markContract();
          continue;
        }

        if (auto cmpxchg = dyn_cast<LLVM::AtomicCmpXchgOp>(&op)) {
          contract.kind = LLVMSafetyEffectKind::AtomicCmpXchg;
          markContract();
          continue;
        }

        if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
          if (load.getOrdering() == LLVM::AtomicOrdering::not_atomic)
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicLoad;
          contract.ordering = mapAtomicOrdering(load.getOrdering());
          markContract();
          continue;
        }

        if (auto store = dyn_cast<LLVM::StoreOp>(&op)) {
          if (store.getOrdering() == LLVM::AtomicOrdering::not_atomic)
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicStore;
          contract.ordering = mapAtomicOrdering(store.getOrdering());
          markContract();
        }
      }
    }
  });
}

static void setLythonSafetyMetadataId(llvm::Instruction &inst, int64_t id) {
  llvm::LLVMContext &ctx = inst.getContext();
  llvm::Metadata *operands[] = {
      llvm::MDString::get(ctx, kLythonSafetyMetadataVersion),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), id))};
  inst.setMetadata(kLythonSafetyMetadataName, llvm::MDNode::get(ctx, operands));
}

static std::optional<int64_t>
getLythonSafetyMetadataId(llvm::Instruction &inst) {
  llvm::MDNode *node = inst.getMetadata(kLythonSafetyMetadataName);
  if (!node || node->getNumOperands() != 2)
    return std::nullopt;
  auto *version = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
  if (!version || version->getString() != kLythonSafetyMetadataVersion)
    return std::nullopt;
  auto *constant =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(node->getOperand(1));
  if (!constant)
    return std::nullopt;
  auto *intConstant = llvm::dyn_cast<llvm::ConstantInt>(constant->getValue());
  if (!intConstant)
    return std::nullopt;
  return intConstant->getSExtValue();
}

void collectLinkedLLVMSafetyContracts(llvm::Module &llvmModule,
                                      LLVMSafetyProfile &profile) {
  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        if (getLythonSafetyMetadataId(inst))
          continue;

        LLVMSafetyContract contract;
        contract.id = static_cast<int64_t>(profile.contracts.size());
        contract.functionName = function.getName().str();

        if (auto *rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&inst)) {
          contract.kind = LLVMSafetyEffectKind::AtomicRMW;
          contract.rmwBinOp = rmw->getOperation();
          contract.integerOperand = llvmIntegerConstant(rmw->getValOperand());
          contract.ordering = rmw->getOrdering();
        } else if (llvm::isa<llvm::AtomicCmpXchgInst>(inst)) {
          contract.kind = LLVMSafetyEffectKind::AtomicCmpXchg;
        } else if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
          if (!load->isAtomic())
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicLoad;
          contract.ordering = load->getOrdering();
        } else if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
          if (!store->isAtomic())
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicStore;
          contract.ordering = store->getOrdering();
        } else {
          continue;
        }

        setLythonSafetyMetadataId(inst, contract.id);
        profile.contracts.push_back(std::move(contract));
      }
    }
  }
}

class PySafetyLLVMIRTranslationInterface final
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult amendOperation(Operation *op,
                               ArrayRef<llvm::Instruction *> instructions,
                               NamedAttribute attribute,
                               LLVM::ModuleTranslation &) const final {
    if (attribute.getName() != kPySafetyContractIdAttr)
      return success();
    auto idAttr = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!idAttr)
      return op->emitOpError("py.safety_contract_id must be an integer");
    int64_t id = idAttr.getInt();
    for (llvm::Instruction *instruction : instructions)
      setLythonSafetyMetadataId(*instruction, id);
    return success();
  }
};

void registerPySafetyLLVMIRTranslation(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, py::PyDialect *dialect) {
    dialect->addInterfaces<PySafetyLLVMIRTranslationInterface>();
  });
}

static void emitLLVMSafetyVerifierError(llvm::Instruction &inst,
                                        llvm::StringRef msg,
                                        llvm::raw_ostream &diag);

static LogicalResult verifyLLVMIRSafetyMetadataPreserved(
    llvm::Module &llvmModule, const LLVMSafetyProfile &profile,
    llvm::StringRef label, LLVMSafetyContractCoverage coverage,
    llvm::raw_ostream &diag) {
  std::vector<unsigned> used(profile.contracts.size(), 0);
  std::vector<unsigned> reserved(profile.contracts.size(), 0);
  bool failedAny = false;

  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        if (auto id = getLythonSafetyMetadataId(inst))
          if (*id >= 0 && static_cast<size_t>(*id) < reserved.size())
            ++reserved[static_cast<size_t>(*id)];
      }
    }
  }

  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto id = getLythonSafetyMetadataId(inst);
        if (!id) {
          if (auto recovered =
                  recoverDroppedSafetyContract(inst, profile, reserved)) {
            setLythonSafetyMetadataId(inst, *recovered);
            id = recovered;
            if (*recovered >= 0 &&
                static_cast<size_t>(*recovered) < reserved.size())
              ++reserved[static_cast<size_t>(*recovered)];
          } else if (getStructuralSafetyEffectKind(inst)) {
            emitLLVMSafetyVerifierError(
                inst,
                "LLVM atomic safety effect has no preserved MLIR "
                "contract id",
                diag);
            failedAny = true;
          }
          if (!id)
            continue;
        }
        if (*id < 0 || static_cast<size_t>(*id) >= profile.contracts.size()) {
          emitLLVMSafetyVerifierError(
              inst, "LLVM IR safety effect has no preserved MLIR contract id",
              diag);
          failedAny = true;
          continue;
        }

        const LLVMSafetyContract &contract =
            profile.contracts[static_cast<size_t>(*id)];
        if (!instructionMatchesContract(inst, contract)) {
          emitLLVMSafetyVerifierError(
              inst,
              "LLVM IR safety effect shape differs from MLIR contract "
              "id",
              diag);
          failedAny = true;
          continue;
        }
        ++used[static_cast<size_t>(*id)];
      }
    }
  }

  if (coverage == LLVMSafetyContractCoverage::RequireEveryContract) {
    for (auto indexed : llvm::enumerate(used)) {
      if (indexed.value() != 0)
        continue;
      const LLVMSafetyContract &contract = profile.contracts[indexed.index()];
      diag << "error: " << label << ": MLIR safety contract for @"
           << contract.functionName << " was not preserved"
           << " (kind=" << static_cast<int>(contract.kind);
      if (contract.rmwBinOp)
        diag << ", rmw=" << static_cast<int>(*contract.rmwBinOp);
      if (contract.integerOperand)
        diag << ", value=" << *contract.integerOperand;
      if (contract.ordering)
        diag << ", ordering=" << static_cast<int>(*contract.ordering);
      diag << ")\n";
      failedAny = true;
    }
  }

  return failure(failedAny);
}

LogicalResult
verifyLLVMIRSafetyMetadataAttached(llvm::Module &llvmModule,
                                   const LLVMSafetyProfile &profile,
                                   llvm::raw_ostream &diag) {
  return verifyLLVMIRSafetyMetadataPreserved(
      llvmModule, profile, "LLVM IR safety metadata verifier",
      LLVMSafetyContractCoverage::RequireEveryContract, diag);
}

static void emitLLVMSafetyVerifierError(llvm::Instruction &inst,
                                        llvm::StringRef msg,
                                        llvm::raw_ostream &diag) {
  diag << "error: LLVM safety verifier: " << msg << "\n";
  if (llvm::Function *function = inst.getFunction())
    diag << "  in function: " << function->getName() << "\n";
  diag << "  instruction: " << inst << "\n";
}

LogicalResult verifyPostCoroLLVMThreadSafe(llvm::Module &llvmModule,
                                           const LLVMSafetyProfile &profile,
                                           llvm::raw_ostream &diag,
                                           LLVMSafetyContractCoverage coverage) {
  if (llvm::verifyModule(llvmModule, &diag))
    return failure();

  return verifyLLVMIRSafetyMetadataPreserved(
      llvmModule, profile, "post-coro LLVM safety verifier", coverage, diag);
}

LogicalResult verifyOptimizedLLVMThreadSafe(llvm::Module &llvmModule,
                                            const LLVMSafetyProfile &profile,
                                            llvm::raw_ostream &diag) {
  return verifyPostCoroLLVMThreadSafe(
      llvmModule, profile, diag,
      LLVMSafetyContractCoverage::AllowOptimizerElision);
}

} // namespace lython::driver
