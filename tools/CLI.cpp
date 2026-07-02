#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Error.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/X86TargetParser.h"
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "llvm/Transforms/Coroutines/CoroEarly.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

#include <sys/resource.h>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "Emitter.h"
#include "Parser.h"
#include "Passes/Runtime/Cleanup.h"
#include "Passes/RuntimeLowering/Arch/Arm/PrimitiveTensorArmSME.h"
#include "Passes/RuntimeLowering/Arch/X86/PrimitiveTensorX86.h"
#include "embedded.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

// Forward declare our custom lowering pass factories
namespace py {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPyOptimizationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncThunkLoweringPass();
} // namespace py

namespace {
std::string TargetTriple;
std::string TargetCPU;
std::string TargetFPU;
std::string TargetFloatABI;
std::string TargetSysroot;
std::vector<std::string> IncludeSearchPaths;
std::vector<std::string> LibrarySearchPaths;

std::string trimEnvToken(llvm::StringRef token) {
  token = token.trim();
  return token.str();
}

std::string configuredTargetTripleOverride() {
  return trimEnvToken(TargetTriple);
}

std::string configuredTargetSysrootOverride() {
  return trimEnvToken(TargetSysroot);
}

bool parseConfiguredFloatABI(llvm::FloatABI::ABIType &result) {
  llvm::StringRef value = TargetFloatABI;
  value = value.trim();
  if (value.empty() || value == "default") {
    result = llvm::FloatABI::Default;
    return true;
  }
  if (value == "soft" || value == "softfp") {
    result = llvm::FloatABI::Soft;
    return true;
  }
  if (value == "hard") {
    result = llvm::FloatABI::Hard;
    return true;
  }
  llvm::errs() << "error: unsupported -mfloat-abi value '" << value
               << "'; expected default, soft, softfp, or hard\n";
  return false;
}

struct IRDumpConfig {
  bool all = false;
  std::set<std::string> passes;

  static IRDumpConfig fromEnv() {
    IRDumpConfig config;
    auto value = llvm::sys::Process::GetEnv("LYTHON_IR_DUMP");
    if (!value || value->empty())
      return config;
    llvm::SmallVector<llvm::StringRef, 16> tokens;
    llvm::StringRef(*value).split(tokens, ",", /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
    for (llvm::StringRef token : tokens) {
      std::string name = trimEnvToken(token);
      if (name.empty())
        continue;
      if (name == "all" || name == "*") {
        config.all = true;
        continue;
      }
      config.passes.insert(std::move(name));
    }
    return config;
  }

  bool shouldDump(llvm::StringRef passName) const {
    return all || passes.count(passName.str()) != 0;
  }
};

std::string hostCPUNameForCodeGen() {
  llvm::StringRef cpu = llvm::sys::getHostCPUName();
  if (cpu.empty())
    return "generic";
  return cpu.str();
}

std::string hostCPUFeaturesForCodeGen() {
  llvm::SubtargetFeatures features;
  for (const auto &entry : llvm::sys::getHostCPUFeatures())
    features.AddFeature(entry.getKey(), entry.getValue());
#if defined(__APPLE__) && defined(__aarch64__)
  auto addDarwinArmFeature = [&](llvm::StringRef sysctlName,
                                 llvm::StringRef featureName) {
    int enabled = 0;
    size_t size = sizeof(enabled);
    if (sysctlbyname(sysctlName.str().c_str(), &enabled, &size, nullptr, 0) ==
            0 &&
        enabled != 0)
      features.AddFeature(featureName);
  };
  addDarwinArmFeature("hw.optional.arm.FEAT_SME", "sme");
  addDarwinArmFeature("hw.optional.arm.FEAT_SME2", "sme2");
#endif
  return features.getString();
}

llvm::Triple codeGenTripleForTarget(py::TensorLoweringTarget target) {
  (void)target;
  std::string override = configuredTargetTripleOverride();
  return llvm::Triple(override.empty() ? llvm::sys::getDefaultTargetTriple()
                                       : override);
}

bool targetFeatureEnabled(const llvm::Triple &triple, llvm::StringRef feature);

bool targetFeatureStringContains(llvm::StringRef features,
                                 llvm::StringRef feature) {
  llvm::SmallVector<llvm::StringRef, 32> tokens;
  features.split(tokens, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef token : tokens) {
    token = token.trim();
    token.consume_front("+");
    token.consume_front("-");
    if (token == feature)
      return true;
  }
  return false;
}

void appendTargetFeature(std::string &features, llvm::StringRef feature) {
  if (targetFeatureStringContains(features, feature))
    return;
  if (!features.empty())
    features += ",";
  features += "+";
  features += feature;
}

std::string configuredCPUNameForCodeGen(const llvm::Triple &triple) {
  llvm::StringRef cpu = TargetCPU;
  if (cpu.empty())
    return "";
  if (cpu != "native")
    return cpu.str();

  llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  if (triple.getArch() == hostTriple.getArch())
    return hostCPUNameForCodeGen();
  return "";
}

std::string codeGenCPUNameForTarget(py::TensorLoweringTarget target,
                                    const llvm::Triple &triple) {
  std::string configuredCPU = configuredCPUNameForCodeGen(triple);
  if (!configuredCPU.empty())
    return configuredCPU;
  if (target.usesX86AVX2FMA())
    return "haswell";
  if (target.usesX86SSE42())
    return "nehalem";
  llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  if (triple.getArch() == hostTriple.getArch())
    return hostCPUNameForCodeGen();
  if (triple.getArch() == llvm::Triple::x86_64)
    return "x86-64";
  return hostCPUNameForCodeGen();
}

std::string codeGenFeaturesForTarget(py::TensorLoweringTarget target,
                                     const llvm::Triple &triple) {
  if (target.usesX86AVX2FMA())
    return "+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+fma";
  if (target.usesX86SSE42())
    return "+sse2,+sse3,+ssse3,+sse4.1,+sse4.2";
  if (target.usesArmSME()) {
    std::string features;
    llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
    if (triple.getArch() == hostTriple.getArch())
      features = hostCPUFeaturesForCodeGen();
    appendTargetFeature(features, "sme");
    if (targetFeatureEnabled(triple, "sme2"))
      appendTargetFeature(features, "sme2");
    return features;
  }
  llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  if (triple.getArch() != hostTriple.getArch())
    return "";
  return hostCPUFeaturesForCodeGen();
}

llvm::ExceptionHandling
exceptionModelForTargetTriple(const llvm::Triple &triple) {
  if (triple.isOSWindows())
    return llvm::ExceptionHandling::WinEH;
  return llvm::ExceptionHandling::DwarfCFI;
}

std::unique_ptr<llvm::TargetMachine>
createCodeGenTargetMachine(py::TensorLoweringTarget target,
                           std::string *normalizedTriple = nullptr) {
  llvm::Triple triple = codeGenTripleForTarget(target);
  std::string targetTriple = triple.normalize();
  if (normalizedTriple)
    *normalizedTriple = targetTriple;

  std::string error;
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!llvmTarget) {
    llvm::errs() << "Failed to lookup target: " << error << "\n";
    return nullptr;
  }

  llvm::TargetOptions opt;
  opt.ExceptionModel = exceptionModelForTargetTriple(triple);
  opt.MCOptions.EmitCompactUnwindNonCanonical = true;
  opt.ForceDwarfFrameSection = true;
  opt.MCOptions.EmitDwarfUnwind = llvm::EmitDwarfUnwindType::Always;
  if (!parseConfiguredFloatABI(opt.FloatABIType))
    return nullptr;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      llvmTarget->createTargetMachine(
          targetTriple, codeGenCPUNameForTarget(target, triple),
          codeGenFeaturesForTarget(target, triple), opt, std::nullopt));
  if (!targetMachine)
    llvm::errs() << "Failed to create target machine for " << targetTriple
                 << "\n";
  return targetMachine;
}

bool hostFeatureEnabled(llvm::StringRef feature) {
  auto features = llvm::sys::getHostCPUFeatures();
  auto found = features.find(feature);
  if (found != features.end() && found->second)
    return true;

#if defined(__APPLE__) && defined(__aarch64__)
  llvm::StringRef sysctlName;
  if (feature == "sme")
    sysctlName = "hw.optional.arm.FEAT_SME";
  else if (feature == "sme2")
    sysctlName = "hw.optional.arm.FEAT_SME2";
  if (!sysctlName.empty()) {
    int enabled = 0;
    size_t size = sizeof(enabled);
    return sysctlbyname(sysctlName.str().c_str(), &enabled, &size, nullptr,
                        0) == 0 &&
           enabled != 0;
  }
#endif

  return false;
}

bool isHostCodeGenTriple(const llvm::Triple &triple) {
  return triple.normalize() ==
         llvm::Triple(llvm::sys::getDefaultTargetTriple()).normalize();
}

bool featureNameMatches(llvm::StringRef value, llvm::StringRef feature) {
  value.consume_front("+");
  return value == feature;
}

bool x86CPUFeatureEnabled(llvm::StringRef cpu, llvm::StringRef feature) {
  llvm::SmallVector<llvm::StringRef, 64> features;
  llvm::X86::getFeaturesForCPU(cpu, features, /*NeedPlus=*/false);
  return llvm::is_contained(features, feature);
}

bool aarch64CPUFeatureEnabled(llvm::StringRef cpu, llvm::StringRef feature) {
  std::optional<llvm::AArch64::CpuInfo> parsed = llvm::AArch64::parseCpu(cpu);
  if (!parsed)
    return false;
  llvm::AArch64::ExtensionSet extensions;
  extensions.addCPUDefaults(*parsed);
  std::vector<std::string> features;
  extensions.toLLVMFeatureList(features);
  return llvm::any_of(features, [&](llvm::StringRef value) {
    return featureNameMatches(value, feature);
  });
}

bool targetFeatureEnabled(const llvm::Triple &triple, llvm::StringRef feature) {
  llvm::StringRef cpu = TargetCPU;
  if (!cpu.empty() && cpu != "native") {
    if (triple.getArch() == llvm::Triple::x86_64)
      return x86CPUFeatureEnabled(cpu, feature);
    if (triple.isAArch64())
      return aarch64CPUFeatureEnabled(cpu, feature);
    return false;
  }

  if (!isHostCodeGenTriple(triple) && cpu != "native")
    return false;
  return hostFeatureEnabled(feature);
}

py::TensorLoweringTarget detectTensorLoweringTarget(llvm::Triple triple) {
  py::TensorLoweringTarget target;
  if (triple.isAArch64() && (targetFeatureEnabled(triple, "sme") ||
                             targetFeatureEnabled(triple, "sme2")))
    target.architecture = py::TensorLoweringArchitecture::ArmSME;
  if (triple.getArch() == llvm::Triple::x86_64) {
    if (targetFeatureEnabled(triple, "avx2") &&
        targetFeatureEnabled(triple, "fma"))
      target.architecture = py::TensorLoweringArchitecture::X86AVX2FMA;
    else if (targetFeatureEnabled(triple, "sse4.2"))
      target.architecture = py::TensorLoweringArchitecture::X86SSE42;
  }
  return target;
}

void dumpMLIRForPass(const IRDumpConfig &config, llvm::StringRef passName,
                     ModuleOp module) {
  if (!config.shouldDump(passName))
    return;
  llvm::errs() << "\n=== [LYTHON_IR_DUMP:" << passName << " MLIR] ===\n";
  module->print(llvm::errs());
  llvm::errs() << "\n";
}

void dumpLLVMForPass(const IRDumpConfig &config, llvm::StringRef passName,
                     llvm::Module &module) {
  if (!config.shouldDump(passName))
    return;
  llvm::errs() << "\n=== [LYTHON_IR_DUMP:" << passName << " LLVM] ===\n";
  module.print(llvm::errs(), nullptr);
  llvm::errs() << "\n";
}

bool perfEnabled() {
  static const bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_PERF");
    if (!value)
      return false;
    llvm::StringRef text(*value);
    return text == "1" || text.equals_insensitive("true") ||
           text.equals_insensitive("yes") || text.equals_insensitive("on");
  }();
  return enabled;
}

std::uint64_t timevalMicros(const timeval &value) {
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000ULL +
         static_cast<std::uint64_t>(value.tv_usec);
}

void *armStreamingCompatibleMemcpy(void *dest, const void *src,
                                   std::size_t count) {
  auto *out = static_cast<volatile unsigned char *>(dest);
  const auto *in = static_cast<const volatile unsigned char *>(src);
  for (std::size_t index = 0; index < count; ++index)
    out[index] = in[index];
  return dest;
}

void *armStreamingCompatibleMemmove(void *dest, const void *src,
                                    std::size_t count) {
  auto *out = static_cast<volatile unsigned char *>(dest);
  const auto *in = static_cast<const volatile unsigned char *>(src);
  if (reinterpret_cast<std::uintptr_t>(dest) <=
      reinterpret_cast<std::uintptr_t>(src)) {
    for (std::size_t index = 0; index < count; ++index)
      out[index] = in[index];
  } else {
    for (std::size_t index = count; index > 0; --index)
      out[index - 1] = in[index - 1];
  }
  return dest;
}

void *armStreamingCompatibleMemset(void *dest, int value, std::size_t count) {
  auto *out = static_cast<volatile unsigned char *>(dest);
  auto byte = static_cast<unsigned char>(value);
  for (std::size_t index = 0; index < count; ++index)
    out[index] = byte;
  return dest;
}

void *armStreamingCompatibleMemchr(void *ptr, int value, std::size_t count) {
  auto *bytes = static_cast<unsigned char *>(ptr);
  auto needle = static_cast<unsigned char>(value);
  for (std::size_t index = 0; index < count; ++index) {
    volatile unsigned char current = bytes[index];
    if (current == needle)
      return bytes + index;
  }
  return nullptr;
}

#if defined(__linux__)
long perfEventOpen(perf_event_attr *attr, pid_t pid, int cpu, int groupFd,
                   unsigned long flags) {
  return syscall(__NR_perf_event_open, attr, pid, cpu, groupFd, flags);
}

class HardwareCounter {
public:
  explicit HardwareCounter(std::uint64_t config) {
    perf_event_attr attr = {};
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv = 1;
    fd = static_cast<int>(perfEventOpen(&attr, /*pid=*/0, /*cpu=*/-1,
                                        /*groupFd=*/-1, /*flags=*/0));
  }

  ~HardwareCounter() {
    if (fd >= 0)
      close(fd);
  }

  void start() {
    if (fd < 0)
      return;
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  }

  std::optional<std::uint64_t> stop() {
    if (fd < 0)
      return std::nullopt;
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    std::uint64_t value = 0;
    if (read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value)))
      return std::nullopt;
    return value;
  }

private:
  int fd = -1;
};
#endif

class PerfScope {
public:
  explicit PerfScope(llvm::StringRef phase) : phase(phase.str()) {
    if (!perfEnabled())
      return;
    enabled = true;
    wallStart = Clock::now();
    getrusage(RUSAGE_SELF, &usageStart);
#if defined(__linux__)
    instructions.emplace(PERF_COUNT_HW_INSTRUCTIONS);
    cycles.emplace(PERF_COUNT_HW_CPU_CYCLES);
    instructions->start();
    cycles->start();
#endif
  }

  ~PerfScope() {
    if (!enabled)
      return;
#if defined(__linux__)
    std::optional<std::uint64_t> instructionCount = instructions->stop();
    std::optional<std::uint64_t> cycleCount = cycles->stop();
#else
    std::optional<std::uint64_t> instructionCount;
    std::optional<std::uint64_t> cycleCount;
#endif
    rusage usageEnd = {};
    getrusage(RUSAGE_SELF, &usageEnd);
    auto wallEnd = Clock::now();
    auto wallUs = std::chrono::duration_cast<std::chrono::microseconds>(
                      wallEnd - wallStart)
                      .count();
    std::uint64_t userUs =
        timevalMicros(usageEnd.ru_utime) - timevalMicros(usageStart.ru_utime);
    std::uint64_t sysUs =
        timevalMicros(usageEnd.ru_stime) - timevalMicros(usageStart.ru_stime);

    llvm::errs()
        << "[LYTHON_PERF] phase=" << phase << " wall_us=" << wallUs
        << " user_us=" << userUs << " sys_us=" << sysUs
        << " minor_faults=" << (usageEnd.ru_minflt - usageStart.ru_minflt)
        << " major_faults=" << (usageEnd.ru_majflt - usageStart.ru_majflt)
        << " voluntary_csw=" << (usageEnd.ru_nvcsw - usageStart.ru_nvcsw)
        << " involuntary_csw=" << (usageEnd.ru_nivcsw - usageStart.ru_nivcsw)
        << " maxrss=" << usageEnd.ru_maxrss;
    if (instructionCount)
      llvm::errs() << " instructions=" << *instructionCount;
    else
      llvm::errs() << " instructions=unavailable";
    if (cycleCount)
      llvm::errs() << " cycles=" << *cycleCount;
    else
      llvm::errs() << " cycles=unavailable";
    llvm::errs() << "\n";
  }

private:
  using Clock = std::chrono::steady_clock;

  std::string phase;
  bool enabled = false;
  Clock::time_point wallStart;
  rusage usageStart = {};
#if defined(__linux__)
  std::optional<HardwareCounter> instructions;
  std::optional<HardwareCounter> cycles;
#endif
};

template <typename Populate>
LogicalResult runLoweringPhase(llvm::StringRef name, MLIRContext &context,
                               ModuleOp module, Populate populate) {
  std::string phase = ("lowering." + name).str();
  PerfScope perf(phase);
  PassManager pm(&context);
  populate(pm);
  return pm.run(module);
}

LogicalResult requireNoAsyncDialectOps(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "async")
      return WalkResult::advance();
    op->emitError()
        << "unlowered async dialect operation remains after Lython async "
           "runtime lowering; MLIR bundled async runtime is not part of the "
           "runtime model";
    result = failure();
    return WalkResult::interrupt();
  });
  return result;
}

LogicalResult runPipeline(ModuleOp module, MLIRContext &context,
                          const IRDumpConfig &irDump,
                          py::TensorLoweringTarget tensorTarget) {
  dumpMLIRForPass(irDump, "frontend", module);

  // Phase 1: Native verification
  if (failed(runLoweringPhase("native-verification", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createNativeVerificationPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "native-verification", module);

  // Phase 2: Publication preparation
  if (failed(runLoweringPhase(
          "publication-preparation", context, module, [&](PassManager &pm) {
            pm.addPass(py::createPublicationPreparationPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "publication-preparation", module);

  // Phase 3.15: High-level Py semantic optimizations. This must run before
  // runtime MLIR embedding so the imported runtime roots match the optimized
  // py dialect, and before OwnershipVerifierPass so all ownership rewrites are
  // still checked by the quantitative kernel.
  if (failed(runLoweringPhase("py-optimization", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createPyOptimizationPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "py-optimization", module);

  // Phase 3.2: Quantitative ownership verification
  if (failed(runLoweringPhase("ownership-verifier", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createOwnershipVerifierPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "ownership-verifier", module);

  // Phase 4: Early canonicalization and CSE
  if (failed(runLoweringPhase("canonicalize", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(mlir::createCanonicalizerPass());
                                pm.addPass(mlir::createCSEPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "canonicalize", module);

  // Phase 4.1: Lower primitive tensor/linalg computations while the high-level
  // contraction structure is still visible.  This keeps Python object lowering
  // separate from numeric kernels and gives affine/vector passes an alias-free
  // memref view of primitive tensors.
  if (failed(runLoweringPhase(
          "linalg-lowering", context, module, [&](PassManager &pm) {
            pm.addPass(py::createLinalgLoweringPass(tensorTarget));
          })))
    return failure();
  dumpMLIRForPass(irDump, "linalg-lowering", module);

  // Phase 4.5: Import runtime object definitions written in MLIR. These
  // fragments are kept at func/memref level and flow through the same lowering
  // pipeline as user code.
  {
    PerfScope perf("lowering.runtime-objects");
    if (failed(py::runtime_library::embedObjectModules(module)))
      return failure();
  }
  dumpMLIRForPass(irDump, "runtime-objects", module);

  // Phase 5: Runtime lowering (Py dialect -> func/LLVM)
  if (failed(runLoweringPhase("runtime-lowering", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createRuntimeLoweringPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "runtime-lowering", module);

  // Phase 5.05: Manifest-driven ownership release insertion. This runs after
  // runtime lowering because only then do user operations expose concrete
  // runtime calls and runtime-library deallocator ABI shapes.
  if (failed(runLoweringPhase("refcount-insertion", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createRefCountInsertionPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-insertion", module);

  // Phase 5.06: Proven refcount pair elision.
  if (failed(runLoweringPhase("refcount-elision", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createRefCountPairElisionPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "refcount-elision", module);

  // Phase 5.1: Lower Lython-owned Python await thunks from the MLIR async
  // dialect to the Lython runtime ABI before symbol cleanup can erase the
  // statically selected coroutine body.
  if (failed(runLoweringPhase("async-thunk-lowering", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(py::createAsyncThunkLoweringPass());
                                pm.addPass(mlir::createCanonicalizerPass());
                                pm.addPass(mlir::createCSEPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "async-thunk-lowering", module);
  if (failed(requireNoAsyncDialectOps(module)))
    return failure();

  // Phase 5.5: Let generic MLIR cleanup remove artifacts created by lowering
  // and runtime embedding.
  if (failed(runLoweringPhase("post-lowering-cleanup", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(mlir::createCanonicalizerPass());
                                pm.addPass(mlir::createCSEPass());
                                pm.addPass(mlir::createSymbolDCEPass());
                              })))
    return failure();
  {
    PerfScope perf("lowering.pointer-roundtrip-cleanup");
    while (py::lowering::runtime::cleanup::pointerRoundTrips(module))
      ;
  }
  dumpMLIRForPass(irDump, "post-lowering-cleanup", module);

  // Phase 5.6: Validate that post-lowering cleanup did not alter ownership of
  // runtime calls that return or consume owned object-family descriptors.
  if (failed(runLoweringPhase(
          "llvm-call-verifier", context, module, [&](PassManager &pm) {
            pm.addPass(py::createLLVMCallOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "llvm-call-verifier", module);

  // Phase 5.7: Validate memref-level no-GIL contracts before final lowering.
  if (failed(runLoweringPhase(
          "thread-safety-verifier", context, module, [&](PassManager &pm) {
            pm.addPass(py::createLLVMThreadSafetyVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "thread-safety-verifier", module);

  // Phase 7: Final lowering to LLVM
  {
    py::LoweredSafetyContracts finalSafetyContracts;
    {
      PerfScope perf("lowering.collect-final-safety-contracts");
      py::collectLoweredSafetyContracts(module, finalSafetyContracts);
    }
    if (failed(runLoweringPhase(
            "convert-to-llvm", context, module, [&](PassManager &pm) {
              mlir::ConvertVectorToLLVMPassOptions vectorOptions;
              vectorOptions.reassociateFPReductions = true;
              vectorOptions.x86Vector = tensorTarget.usesX86();
              mlir::VectorTransferToSCFOptions transferOptions;
              transferOptions.setTargetRank(1);
              pm.addPass(mlir::createLowerAffinePass());
              pm.addPass(mlir::memref::createExpandStridedMetadataPass());
              pm.addPass(mlir::createLowerAffinePass());
              if (tensorTarget.usesArmSME())
                py::runtime_lowering::arch::arm::
                    addSMEPreControlFlowLLVMPrepPipeline(pm);
              pm.addNestedPass<mlir::func::FuncOp>(
                  mlir::vector::createLowerVectorMultiReductionPass(
                      mlir::vector::VectorMultiReductionLowering::
                          InnerReduction));
              pm.addPass(mlir::createConvertVectorToSCFPass(transferOptions));
              pm.addPass(mlir::createLowerAffinePass());
              pm.addPass(mlir::createCanonicalizerPass());
              pm.addPass(mlir::createConvertVectorToLLVMPass(vectorOptions));
              pm.addPass(mlir::createConvertSCFToCFPass());
              if (tensorTarget.usesArmSME())
                py::runtime_lowering::arch::arm::
                    addSMEPostControlFlowLLVMPrepPipeline(pm);
              pm.addPass(mlir::createArithToLLVMConversionPass());
              pm.addPass(mlir::createConvertControlFlowToLLVMPass());
              pm.addPass(mlir::createConvertToLLVMPass());
              pm.addPass(mlir::createReconcileUnrealizedCastsPass());
              pm.addNestedPass<mlir::func::FuncOp>(
                  mlir::createReconcileUnrealizedCastsPass());
              pm.addPass(mlir::createCanonicalizerPass());
            })))
      return failure();
    {
      PerfScope perf("lowering.preserve-final-safety-contracts");
      if (failed(
              py::preserveLoweredSafetyContracts(module, finalSafetyContracts)))
        return failure();
    }
    {
      PerfScope perf("lowering.final-llvm-cleanup");
      py::optimizer::pipeline::finalLLVMCleanup(module);
    }
  }
  dumpMLIRForPass(irDump, "convert-to-llvm", module);

  // Phase 7.4: Re-check ownership after final conversion rewrites.
  if (failed(runLoweringPhase(
          "final-ownership-verifier", context, module, [&](PassManager &pm) {
            pm.addPass(py::createLLVMCallOwnershipVerifierPass());
          })))
    return failure();
  dumpMLIRForPass(irDump, "final-ownership-verifier", module);

  // Phase 7.5: Validate final LLVM atomic orderings after MemRefToLLVM.
  if (failed(runLoweringPhase("final-thread-safety-verifier", context, module,
                              [&](PassManager &pm) {
                                pm.addPass(
                                    py::createLLVMThreadSafetyVerifierPass());
                              })))
    return failure();
  dumpMLIRForPass(irDump, "final-thread-safety-verifier", module);

  return success();
}

llvm::orc::SymbolMap
buildRuntimeSymbolMap(llvm::orc::MangleAndInterner interner) {
  llvm::orc::SymbolMap symbolMap;
  auto add = [&](llvm::StringRef name, auto *ptr) {
    symbolMap[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(ptr),
                                 llvm::JITSymbolFlags::Exported};
  };
  add("__arm_sc_memcpy", &armStreamingCompatibleMemcpy);
  add("__arm_sc_memmove", &armStreamingCompatibleMemmove);
  add("__arm_sc_memset", &armStreamingCompatibleMemset);
  add("__arm_sc_memchr", &armStreamingCompatibleMemchr);
  return symbolMap;
}

OwningOpRef<ModuleOp> parseModuleFromBuffer(StringRef buffer,
                                            MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(buffer, &context);
  if (!module)
    llvm::errs() << "error: failed to parse MLIR source\n";
  return module;
}

LogicalResult runCppParserDump(StringRef inputPath, bool typeComments,
                               bool includeAttributes, bool interactiveMode,
                               bool expressionMode, bool functionTypeMode) {
  auto file = llvm::MemoryBuffer::getFile(inputPath);
  if (!file) {
    llvm::errs() << "error: could not open input file '" << inputPath << "'\n";
    return failure();
  }

  lython::parser::ParseOptions options;
  options.typeComments = typeComments;
  if (interactiveMode)
    options.mode = lython::parser::ParseMode::Interactive;
  if (expressionMode)
    options.mode = lython::parser::ParseMode::Expression;
  if (functionTypeMode)
    options.mode = lython::parser::ParseMode::FunctionType;
  lython::parser::ParseResult result;
  {
    PerfScope perf("parse");
    result = lython::parser::parse(file->get()->getBuffer(), inputPath.str(),
                                   options);
  }
  if (!result.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : result.diagnostics) {
      llvm::errs() << inputPath << ':' << diagnostic.location.line << ':'
                   << diagnostic.location.column
                   << ": error: " << diagnostic.message << "\n";
    }
    return failure();
  }

  llvm::outs() << lython::parser::dumpAst(*result.tree, includeAttributes)
               << "\n";
  return success();
}

std::string pythonTracebackPath(StringRef inputPath);

LogicalResult generateModuleFromCppFrontend(StringRef pythonFile,
                                            MLIRContext &context,
                                            OwningOpRef<ModuleOp> &module) {
  auto file = llvm::MemoryBuffer::getFile(pythonFile);
  if (!file) {
    llvm::errs() << "error: could not open input file '" << pythonFile << "'\n";
    return failure();
  }

  lython::parser::ParseOptions options;
  options.typeComments = true;
  lython::parser::ParseResult parsed;
  {
    PerfScope perf("parse");
    parsed = lython::parser::parse(file->get()->getBuffer(), pythonFile.str(),
                                   options);
  }
  if (!parsed.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : parsed.diagnostics) {
      llvm::errs() << pythonFile << ':' << diagnostic.location.line << ':'
                   << diagnostic.location.column
                   << ": parse error: " << diagnostic.message << "\n";
    }
    return failure();
  }

  lython::emitter::EmitResult emitted;
  {
    PerfScope perf("ir-generation");
    emitted = lython::emitter::emitModule(*parsed.tree, context, "__main__",
                                          pythonTracebackPath(pythonFile));
  }
  if (!emitted.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics) {
      llvm::errs() << pythonFile << ':' << diagnostic.location.line << ':'
                   << diagnostic.location.column
                   << ": emit error: " << diagnostic.message << "\n";
    }
    return failure();
  }

  module = std::move(emitted.module);
  return success();
}

std::string pythonTracebackPath(StringRef inputPath) {
  if (llvm::sys::path::is_absolute(inputPath))
    return inputPath.str();
  llvm::SmallString<256> current;
  if (llvm::sys::fs::current_path(current))
    return inputPath.str();
  llvm::sys::path::append(current, inputPath);
  return current.str().str();
}

LogicalResult writeLLVMIR(llvm::Module &llvmModule, StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return failure();
  }
  llvmModule.print(out, nullptr);
  return success();
}

LogicalResult installAOTEntryPoint(llvm::Module &llvmModule) {
  llvm::Function *pythonMain = llvmModule.getFunction("__main__");
  if (!pythonMain) {
    llvm::errs() << "error: cannot build executable: missing __main__ entry\n";
    return failure();
  }
  if (!pythonMain->arg_empty() || pythonMain->isVarArg()) {
    llvm::errs() << "error: cannot build executable: __main__ must not take "
                    "arguments\n";
    return failure();
  }

  if (llvm::Function *existing = llvmModule.getFunction("main")) {
    if (!existing->isDeclaration()) {
      llvm::errs()
          << "error: cannot build executable: symbol 'main' already exists\n";
      return failure();
    }
    existing->eraseFromParent();
  }
  constexpr llvm::StringLiteral kAOTEntryThunkName = "__lython_aot_entry";
  if (llvm::Function *existing = llvmModule.getFunction(kAOTEntryThunkName)) {
    if (!existing->isDeclaration()) {
      llvm::errs() << "error: cannot build executable: symbol '"
                   << kAOTEntryThunkName << "' already exists\n";
      return failure();
    }
    existing->eraseFromParent();
  }

  llvm::LLVMContext &context = llvmModule.getContext();
  llvm::Type *voidTy = llvm::Type::getVoidTy(context);
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *ptr = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *entryThunkType =
      llvm::FunctionType::get(voidTy, /*isVarArg=*/false);
  llvm::Function *entryThunk =
      llvm::Function::Create(entryThunkType, llvm::GlobalValue::InternalLinkage,
                             kAOTEntryThunkName, llvmModule);
  entryThunk->setUWTableKind(llvm::UWTableKind::Async);

  llvm::BasicBlock *thunkBlock =
      llvm::BasicBlock::Create(context, "entry", entryThunk);
  llvm::IRBuilder<> thunkBuilder(thunkBlock);
  thunkBuilder.CreateCall(pythonMain->getFunctionType(), pythonMain, {});
  thunkBuilder.CreateRetVoid();

  llvm::FunctionType *mainType =
      llvm::FunctionType::get(i32, /*isVarArg=*/false);
  llvm::Function *main = llvm::Function::Create(
      mainType, llvm::GlobalValue::ExternalLinkage, "main", llvmModule);
  main->setUWTableKind(llvm::UWTableKind::Async);

  llvm::FunctionType *runnerType =
      llvm::FunctionType::get(i32, {ptr}, /*isVarArg=*/false);
  llvm::FunctionCallee runner =
      llvmModule.getOrInsertFunction("LyRunPythonMain", runnerType);

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", main);
  llvm::IRBuilder<> builder(entry);
  llvm::CallInst *status = builder.CreateCall(runner, {entryThunk});
  builder.CreateRet(status);
  return success();
}

// Lowers LLVM coroutines and runs the standard O2 module pipeline. MLIR-level
// passes never ran SROA/mem2reg-class cleanups on the translated IR, so
// without this the descriptor allocas of every lowered object stay in the
// frame (~2KB per object-handling call frame) and nothing is ever inlined.
// The O2 default pipeline already contains the coroutine lowering phases
// (CoroEarly/CoroSplit/CoroCleanup) at their correct positions.
void runLLVMCoroLowering(llvm::Module &llvmModule,
                         llvm::TargetMachine *targetMachine = nullptr) {
  llvm::LoopAnalysisManager loopAM;
  llvm::FunctionAnalysisManager functionAM;
  llvm::CGSCCAnalysisManager cgsccAM;
  llvm::ModuleAnalysisManager moduleAM;
  llvm::PassBuilder passBuilder(targetMachine);
  passBuilder.registerModuleAnalyses(moduleAM);
  passBuilder.registerCGSCCAnalyses(cgsccAM);
  passBuilder.registerFunctionAnalyses(functionAM);
  passBuilder.registerLoopAnalyses(loopAM);
  passBuilder.crossRegisterProxies(loopAM, functionAM, cgsccAM, moduleAM);

  llvm::ModulePassManager modulePM =
      passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
  modulePM.run(llvmModule, moduleAM);
}

enum class LLVMSafetyEffectKind {
  AtomicRMW,
  AtomicCmpXchg,
  AtomicLoad,
  AtomicStore,
};

struct LLVMSafetyContract {
  int64_t id = -1;
  std::string functionName;
  LLVMSafetyEffectKind kind;
  std::optional<llvm::AtomicRMWInst::BinOp> rmwBinOp;
  std::optional<int64_t> integerOperand;
  std::optional<llvm::AtomicOrdering> ordering;
};

struct LLVMSafetyProfile {
  llvm::SmallVector<LLVMSafetyContract, 64> contracts;
};

enum class LLVMSafetyContractCoverage {
  RequireEveryContract,
  AllowOptimizerElision,
};

static constexpr llvm::StringLiteral kLythonSafetyMetadataName{"ly.safety"};
static constexpr llvm::StringLiteral kLythonSafetyMetadataVersion{
    "ly.safety.v1"};
// Atomics inside the pre-lowered runtime cache were verified by the full
// pipeline when the cache was generated at build time. Their metadata is
// sealed to this version so the per-compile verifier can recognize them
// without confusing their stale contract ids with the current profile.
static constexpr llvm::StringLiteral kLythonRuntimeSafetyMetadataVersion{
    "ly.safety.runtime.v1"};
static constexpr llvm::StringLiteral kPySafetyContractIdAttr{
    "py.safety_contract_id"};

std::optional<LLVMSafetyEffectKind>
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

void setLythonSafetyMetadataId(llvm::Instruction &inst, int64_t id) {
  llvm::LLVMContext &ctx = inst.getContext();
  llvm::Metadata *operands[] = {
      llvm::MDString::get(ctx, kLythonSafetyMetadataVersion),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), id))};
  inst.setMetadata(kLythonSafetyMetadataName, llvm::MDNode::get(ctx, operands));
}

std::optional<int64_t> getLythonSafetyMetadataId(llvm::Instruction &inst) {
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

bool hasRuntimeVerifiedSafetyMetadata(llvm::Instruction &inst) {
  llvm::MDNode *node = inst.getMetadata(kLythonSafetyMetadataName);
  if (!node || node->getNumOperands() < 1)
    return false;
  auto *version = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
  return version && version->getString() == kLythonRuntimeSafetyMetadataVersion;
}

// Rewrites the safety metadata of every contracted instruction to the
// runtime-verified version. Called only by the prelower generation run after
// all verifications have passed; the resulting IR is the build-time cache.
void sealRuntimeSafetyMetadata(llvm::Module &llvmModule) {
  llvm::LLVMContext &ctx = llvmModule.getContext();
  llvm::Metadata *operands[] = {
      llvm::MDString::get(ctx, kLythonRuntimeSafetyMetadataVersion)};
  llvm::MDNode *sealed = llvm::MDNode::get(ctx, operands);
  for (llvm::Function &function : llvmModule)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &inst : block)
        if (getLythonSafetyMetadataId(inst))
          inst.setMetadata(kLythonSafetyMetadataName, sealed);
}

LogicalResult linkPrelinkedRuntime(llvm::Module &llvmModule) {
  std::optional<std::string> path =
      py::runtime_library::prelinkedRuntimeIRPath();
  if (!path)
    return success();
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> runtime =
      llvm::parseIRFile(*path, diagnostic, llvmModule.getContext());
  if (!runtime) {
    llvm::errs() << "error: failed to load pre-lowered runtime IR from '"
                 << *path << "': " << diagnostic.getMessage() << "\n";
    return failure();
  }
  runtime->setDataLayout(llvmModule.getDataLayout());
  runtime->setTargetTriple(llvmModule.getTargetTriple());
  if (llvm::Linker::linkModules(llvmModule, std::move(runtime),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    llvm::errs() << "error: failed to link pre-lowered runtime IR\n";
    return failure();
  }
  return success();
}

bool isPlatformNativeSupport(llvm::StringRef name) {
  return name == "support_darwin" || name == "support_linux" ||
         name == "support_windows";
}

bool shouldLinkEmbeddedLLVMRuntimeModule(llvm::StringRef name,
                                         const llvm::Triple &targetTriple) {
  if (name == "support_darwin")
    return targetTriple.isOSDarwin();
  if (name == "support_linux")
    return targetTriple.isOSLinux();
  if (name == "support_windows")
    return targetTriple.isOSWindows();
  return true;
}

void registerNativeRuntimeDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
}

OwningOpRef<ModuleOp> parseEmbeddedNativeRuntimeModule(
    const py::runtime_library::embedded::Module &entry, MLIRContext &context) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(
          llvm::StringRef(reinterpret_cast<const char *>(entry.data),
                          entry.size),
          entry.name, /*RequiresNullTerminator=*/false),
      llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

LogicalResult lowerNativeRuntimeModule(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return pm.run(module);
}

LogicalResult linkEmbeddedNativeRuntime(llvm::Module &llvmModule) {
  namespace embedded = py::runtime_library::embedded;
  llvm::Triple targetTriple(llvmModule.getTargetTriple());
  bool sawPlatformNativeSupport = false;
  bool linkedPlatformNativeSupport = false;
  DialectRegistry registry;
  registerNativeRuntimeDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (std::size_t index = 0; index < embedded::moduleCount(); ++index) {
    const embedded::Module &entry = embedded::modules()[index];
    if (entry.kind != embedded::ModuleKind::NativeMLIRBytecode)
      continue;
    llvm::StringRef name(entry.name);
    if (isPlatformNativeSupport(name))
      sawPlatformNativeSupport = true;
    if (!shouldLinkEmbeddedLLVMRuntimeModule(name, targetTriple))
      continue;
    if (isPlatformNativeSupport(name))
      linkedPlatformNativeSupport = true;

    OwningOpRef<ModuleOp> nativeModule =
        parseEmbeddedNativeRuntimeModule(entry, context);
    if (!nativeModule) {
      llvm::errs() << "error: failed to parse embedded native runtime MLIR "
                      "bytecode module '"
                   << entry.name << "'\n";
      return failure();
    }
    if (failed(lowerNativeRuntimeModule(*nativeModule))) {
      llvm::errs() << "error: failed to lower embedded native runtime MLIR "
                      "module '"
                   << entry.name << "'\n";
      return failure();
    }
    std::unique_ptr<llvm::Module> runtime =
        mlir::translateModuleToLLVMIR(*nativeModule, llvmModule.getContext());
    if (!runtime) {
      llvm::errs() << "error: failed to translate embedded native runtime MLIR "
                      "module '"
                   << entry.name << "' to LLVM IR\n";
      return failure();
    }
    runtime->setDataLayout(llvmModule.getDataLayout());
    runtime->setTargetTriple(llvmModule.getTargetTriple());
    if (llvm::Linker::linkModules(llvmModule, std::move(runtime))) {
      llvm::errs() << "error: failed to link embedded native runtime module '"
                   << entry.name << "'\n";
      return failure();
    }
  }
  if (sawPlatformNativeSupport && !linkedPlatformNativeSupport) {
    llvm::errs() << "error: no embedded native runtime support for target OS '"
                 << targetTriple.getOSName() << "' in target triple '"
                 << targetTriple.str() << "'\n";
    return failure();
  }
  return success();
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

void emitLLVMSafetyVerifierError(llvm::Instruction &inst, llvm::StringRef msg);

LogicalResult verifyLLVMIRSafetyMetadataPreserved(
    llvm::Module &llvmModule, const LLVMSafetyProfile &profile,
    llvm::StringRef label, LLVMSafetyContractCoverage coverage) {
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
        // Instructions from the pre-lowered runtime cache carry sealed
        // metadata: their contracts were verified when the cache was built.
        if (hasRuntimeVerifiedSafetyMetadata(inst))
          continue;
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
                inst, "LLVM atomic safety effect has no preserved MLIR "
                      "contract id");
            failedAny = true;
          }
          if (!id)
            continue;
        }
        if (*id < 0 || static_cast<size_t>(*id) >= profile.contracts.size()) {
          emitLLVMSafetyVerifierError(
              inst, "LLVM IR safety effect has no preserved MLIR contract id");
          failedAny = true;
          continue;
        }

        const LLVMSafetyContract &contract =
            profile.contracts[static_cast<size_t>(*id)];
        if (!instructionMatchesContract(inst, contract)) {
          emitLLVMSafetyVerifierError(
              inst, "LLVM IR safety effect shape differs from MLIR contract "
                    "id");
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
      llvm::errs() << "error: " << label << ": MLIR safety contract for @"
                   << contract.functionName << " was not preserved"
                   << " (kind=" << static_cast<int>(contract.kind);
      if (contract.rmwBinOp)
        llvm::errs() << ", rmw=" << static_cast<int>(*contract.rmwBinOp);
      if (contract.integerOperand)
        llvm::errs() << ", value=" << *contract.integerOperand;
      if (contract.ordering)
        llvm::errs() << ", ordering=" << static_cast<int>(*contract.ordering);
      llvm::errs() << ")\n";
      failedAny = true;
    }
  }

  return failure(failedAny);
}

LogicalResult
verifyLLVMIRSafetyMetadataAttached(llvm::Module &llvmModule,
                                   const LLVMSafetyProfile &profile) {
  return verifyLLVMIRSafetyMetadataPreserved(
      llvmModule, profile, "LLVM IR safety metadata verifier",
      LLVMSafetyContractCoverage::RequireEveryContract);
}

void emitLLVMSafetyVerifierError(llvm::Instruction &inst, llvm::StringRef msg) {
  llvm::errs() << "error: LLVM safety verifier: " << msg << "\n";
  if (llvm::Function *function = inst.getFunction())
    llvm::errs() << "  in function: " << function->getName() << "\n";
  llvm::errs() << "  instruction: " << inst << "\n";
}

LogicalResult verifyPostCoroLLVMThreadSafety(
    llvm::Module &llvmModule, const LLVMSafetyProfile &profile,
    LLVMSafetyContractCoverage coverage =
        LLVMSafetyContractCoverage::RequireEveryContract) {
  if (llvm::verifyModule(llvmModule, &llvm::errs()))
    return failure();

  return verifyLLVMIRSafetyMetadataPreserved(
      llvmModule, profile, "post-coro LLVM safety verifier", coverage);
}

LogicalResult
verifyOptimizedLLVMThreadSafety(llvm::Module &llvmModule,
                                const LLVMSafetyProfile &profile) {
  return verifyPostCoroLLVMThreadSafety(
      llvmModule, profile, LLVMSafetyContractCoverage::AllowOptimizerElision);
}

LogicalResult emitObjectFile(llvm::Module &llvmModule,
                             const LLVMSafetyProfile &safetyProfile,
                             py::TensorLoweringTarget tensorTarget,
                             StringRef objectPath) {
  std::string targetTriple;
  auto targetMachine = createCodeGenTargetMachine(tensorTarget, &targetTriple);
  if (!targetMachine)
    return failure();
  llvmModule.setTargetTriple(targetTriple);
  llvmModule.setDataLayout(targetMachine->createDataLayout());
  runLLVMCoroLowering(llvmModule, targetMachine.get());
  if (failed(verifyOptimizedLLVMThreadSafety(llvmModule, safetyProfile)))
    return failure();

  std::error_code ec;
  llvm::raw_fd_ostream dest(objectPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Failed to open object file: " << ec.message() << "\n";
    return failure();
  }

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target machine cannot emit object file\n";
    return failure();
  }
  pass.run(llvmModule);
  return success();
}

LogicalResult
configureLLVMModuleCodeGenTarget(llvm::Module &llvmModule,
                                 py::TensorLoweringTarget tensorTarget) {
  std::string targetTriple;
  auto targetMachine = createCodeGenTargetMachine(tensorTarget, &targetTriple);
  if (!targetMachine)
    return failure();

  llvmModule.setTargetTriple(targetTriple);
  llvmModule.setDataLayout(targetMachine->createDataLayout());
  return success();
}

llvm::StringRef exceptionPersonalityForTarget(const llvm::Triple &triple) {
  if (triple.isWindowsGNUEnvironment())
    return "__gxx_personality_seh0";
  return "__gxx_personality_v0";
}

void rewriteExceptionPersonalityForTarget(llvm::Module &llvmModule) {
  llvm::Triple triple(llvmModule.getTargetTriple());
  llvm::StringRef personalityName = exceptionPersonalityForTarget(triple);
  if (personalityName == "__gxx_personality_v0")
    return;

  llvm::LLVMContext &context = llvmModule.getContext();
  llvm::FunctionType *personalityType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(context), /*isVarArg=*/true);
  llvm::Function *personalityFn = llvmModule.getFunction(personalityName);
  if (!personalityFn)
    personalityFn = llvm::Function::Create(personalityType,
                                           llvm::GlobalValue::ExternalLinkage,
                                           personalityName, llvmModule);

  if (llvm::Function *itanium = llvmModule.getFunction("__gxx_personality_v0"))
    itanium->replaceAllUsesWith(personalityFn);
}

enum class LinkerDriverFlavor {
  Clang,
  MinGWGcc,
};

struct LinkerDriver {
  std::string program;
  LinkerDriverFlavor flavor = LinkerDriverFlavor::Clang;
};

std::optional<std::string> findExecutableProgram(llvm::StringRef name) {
  if (auto program = llvm::sys::findProgramByName(name))
    return *program;
  return std::nullopt;
}

std::optional<std::string> findHomebrewMinGWProgram(llvm::StringRef name) {
  const char *prefixes[] = {"/opt/homebrew/opt/mingw-w64/bin",
                            "/usr/local/opt/mingw-w64/bin"};
  for (const char *prefix : prefixes) {
    llvm::SmallString<256> path(prefix);
    llvm::sys::path::append(path, name);
    if (llvm::sys::fs::can_execute(path))
      return path.str().str();
  }
  return std::nullopt;
}

std::optional<std::string> findMinGWLinkerDriver(const llvm::Triple &triple) {
  llvm::StringRef programName;
  if (triple.getArch() == llvm::Triple::x86_64)
    programName = "x86_64-w64-mingw32-g++";
  else if (triple.getArch() == llvm::Triple::x86)
    programName = "i686-w64-mingw32-g++";
  else
    return std::nullopt;

  if (auto program = findExecutableProgram(programName))
    return program;
  return findHomebrewMinGWProgram(programName);
}

std::optional<LinkerDriver>
findExecutableLinkerDriver(py::TensorLoweringTarget tensorTarget) {
  llvm::Triple targetTriple = codeGenTripleForTarget(tensorTarget);
  if (targetTriple.isWindowsGNUEnvironment()) {
    if (auto mingw = findMinGWLinkerDriver(targetTriple))
      return LinkerDriver{*mingw, LinkerDriverFlavor::MinGWGcc};
  }

  auto clangExe = llvm::sys::findProgramByName("clang++");
  if (!clangExe)
    clangExe = llvm::sys::findProgramByName("clang");
  if (!clangExe) {
    llvm::errs() << "error: clang++/clang executable not found in PATH\n";
    return std::nullopt;
  }
  return LinkerDriver{*clangExe, LinkerDriverFlavor::Clang};
}

void appendLinkTargetArgs(std::vector<std::string> &args,
                          py::TensorLoweringTarget tensorTarget,
                          LinkerDriverFlavor driverFlavor) {
  llvm::Triple targetTriple = codeGenTripleForTarget(tensorTarget);
  llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  std::string sysroot = configuredTargetSysrootOverride();
  if (driverFlavor == LinkerDriverFlavor::Clang &&
      targetTriple.normalize() != hostTriple.normalize()) {
    args.emplace_back("-target");
    args.emplace_back(targetTriple.normalize());
  }
  if (!sysroot.empty())
    args.emplace_back("--sysroot=" + sysroot);
  if (driverFlavor == LinkerDriverFlavor::Clang && !TargetCPU.empty())
    args.emplace_back("-mcpu=" + TargetCPU);
  if (driverFlavor == LinkerDriverFlavor::Clang && !TargetFPU.empty())
    args.emplace_back("-mfpu=" + TargetFPU);
  if (driverFlavor == LinkerDriverFlavor::Clang && !TargetFloatABI.empty())
    args.emplace_back("-mfloat-abi=" + TargetFloatABI);
  for (const std::string &path : IncludeSearchPaths)
    args.emplace_back("-I" + path);
  for (const std::string &path : LibrarySearchPaths)
    args.emplace_back("-L" + path);
  if (driverFlavor == LinkerDriverFlavor::Clang &&
      targetTriple.getOS() != hostTriple.getOS())
    args.emplace_back("-fuse-ld=lld");
  if (targetTriple.isWindowsGNUEnvironment()) {
    args.emplace_back("-static");
    args.emplace_back("-static-libstdc++");
    args.emplace_back("-static-libgcc");
  }
  if (tensorTarget.usesX86AVX2FMA()) {
    args.emplace_back("-mavx2");
    args.emplace_back("-mfma");
  } else if (tensorTarget.usesX86SSE42()) {
    args.emplace_back("-msse4.2");
  }
}

void appendLinkTargetLibraries(std::vector<std::string> &args,
                               py::TensorLoweringTarget tensorTarget) {
  llvm::Triple targetTriple = codeGenTripleForTarget(tensorTarget);
  if (!targetTriple.isWindowsGNUEnvironment())
    return;
  args.emplace_back("-Wl,-Bstatic");
  args.emplace_back("-lwinpthread");
}

LogicalResult runLinkerCommand(StringRef clangProgram,
                               const std::vector<std::string> &argStorage,
                               StringRef failureMessage) {
  llvm::SmallVector<llvm::StringRef, 16> args;
  for (const auto &arg : argStorage)
    args.push_back(arg);

  std::string errorMessage;
  bool executionFailed = false;
  int result =
      llvm::sys::ExecuteAndWait(clangProgram, args, std::nullopt, std::nullopt,
                                0, 0, &errorMessage, &executionFailed);
  if (result != 0 || executionFailed) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    llvm::errs() << failureMessage << "\n";
    return failure();
  }
  return success();
}

LogicalResult linkExecutable(StringRef objectPath,
                             py::TensorLoweringTarget tensorTarget,
                             StringRef outputPath) {
  std::optional<LinkerDriver> linker = findExecutableLinkerDriver(tensorTarget);
  if (!linker)
    return failure();

  std::vector<std::string> argStorage;
  argStorage.push_back(linker->program);
  appendLinkTargetArgs(argStorage, tensorTarget, linker->flavor);
  argStorage.emplace_back(objectPath.str());
  appendLinkTargetLibraries(argStorage, tensorTarget);
  argStorage.emplace_back("-O2");
  if (codeGenTripleForTarget(tensorTarget).isOSLinux())
    argStorage.emplace_back("-no-pie");
  argStorage.emplace_back("-o");
  argStorage.emplace_back(outputPath.str());

  return runLinkerCommand(linker->program, argStorage, "error: linking failed");
}

std::optional<FileLineColLoc> findPythonSourceLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    if (fileLoc.getFilename().getValue().ends_with(".py"))
      return fileLoc;
    return std::nullopt;
  }
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return findPythonSourceLoc(nameLoc.getChildLoc());
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations())
      if (auto found = findPythonSourceLoc(child))
        return found;
  }
  return std::nullopt;
}

struct PythonDebugScopeCache {
  MLIRContext *context;
  llvm::StringMap<LLVM::DIFileAttr> files;
  llvm::StringMap<LLVM::DICompileUnitAttr> compileUnits;

  explicit PythonDebugScopeCache(MLIRContext *context) : context(context) {}

  LLVM::DIFileAttr fileFor(StringRef sourcePath) {
    if (auto found = files.find(sourcePath); found != files.end())
      return found->second;

    StringRef directory = llvm::sys::path::parent_path(sourcePath);
    StringRef basename = llvm::sys::path::filename(sourcePath);
    if (directory.empty())
      directory = ".";
    LLVM::DIFileAttr file = LLVM::DIFileAttr::get(context, basename, directory);
    files[sourcePath] = file;
    return file;
  }

  LLVM::DICompileUnitAttr compileUnitFor(StringRef sourcePath) {
    if (auto found = compileUnits.find(sourcePath); found != compileUnits.end())
      return found->second;

    LLVM::DICompileUnitAttr unit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)),
        llvm::dwarf::DW_LANG_Python, fileFor(sourcePath),
        StringAttr::get(context, "lython"),
        /*isOptimized=*/true, LLVM::DIEmissionKind::LineTablesOnly);
    compileUnits[sourcePath] = unit;
    return unit;
  }
};

Location scopedPythonDebugLoc(Location loc, LLVM::DISubprogramAttr scope) {
  if (loc->findInstanceOf<FusedLocWith<LLVM::DILocalScopeAttr>>())
    return loc;
  return FusedLoc::get(loc.getContext(), {loc}, scope);
}

void attachPythonDebugInfo(ModuleOp module) {
  PythonDebugScopeCache cache(module.getContext());
  LLVM::DINullTypeAttr voidType =
      LLVM::DINullTypeAttr::get(module.getContext());
  LLVM::DISubroutineTypeAttr subroutineType = LLVM::DISubroutineTypeAttr::get(
      module.getContext(), ArrayRef<LLVM::DITypeAttr>{voidType});

  module.walk([&](LLVM::LLVMFuncOp function) {
    if (function.getLoc()
            ->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    std::optional<FileLineColLoc> sourceLoc =
        findPythonSourceLoc(function.getLoc());
    if (!sourceLoc)
      return;

    StringRef sourcePath = sourceLoc->getFilename().getValue();
    LLVM::DIFileAttr file = cache.fileFor(sourcePath);
    LLVM::DICompileUnitAttr compileUnit = cache.compileUnitFor(sourcePath);
    StringRef linkageName = function.getSymName();
    StringRef displayName =
        linkageName == "__main__" ? "<module>" : linkageName;
    uint32_t flagBits =
        static_cast<uint32_t>(LLVM::DISubprogramFlags::Definition) |
        static_cast<uint32_t>(LLVM::DISubprogramFlags::Optimized);
    if (linkageName == "__main__")
      flagBits |=
          static_cast<uint32_t>(LLVM::DISubprogramFlags::MainSubprogram);
    auto flags = static_cast<LLVM::DISubprogramFlags>(flagBits);

    LLVM::DISubprogramAttr subprogram = LLVM::DISubprogramAttr::get(
        module.getContext(),
        DistinctAttr::create(UnitAttr::get(module.getContext())), compileUnit,
        compileUnit, StringAttr::get(module.getContext(), displayName),
        StringAttr::get(module.getContext(), linkageName), file,
        sourceLoc->getLine(), sourceLoc->getLine(), flags, subroutineType,
        ArrayRef<LLVM::DINodeAttr>{}, ArrayRef<LLVM::DINodeAttr>{});

    function->setLoc(scopedPythonDebugLoc(function.getLoc(), subprogram));
    function.walk([&](Operation *op) {
      if (op == function.getOperation())
        return;
      if (!findPythonSourceLoc(op->getLoc()))
        return;
      op->setLoc(scopedPythonDebugLoc(op->getLoc(), subprogram));
    });
  });
}

LogicalResult buildExecutable(llvm::Module &llvmModule,
                              const LLVMSafetyProfile &safetyProfile,
                              py::TensorLoweringTarget tensorTarget,
                              StringRef outputPath) {
  llvm::SmallString<256> objectPath;
  if (auto ec =
          llvm::sys::fs::createTemporaryFile("lython", ".o", objectPath)) {
    llvm::errs() << "error: failed to create temporary object file: "
                 << ec.message() << "\n";
    return failure();
  }
  llvm::FileRemover objCleanup(objectPath);
  if (failed(
          emitObjectFile(llvmModule, safetyProfile, tensorTarget, objectPath)))
    return failure();
  if (failed(linkExecutable(objectPath, tensorTarget, outputPath)))
    return failure();
  return success();
}

LogicalResult runJIT(ModuleOp module, const IRDumpConfig &irDump,
                     py::TensorLoweringTarget tensorTarget) {
  llvm::Triple processTriple(llvm::sys::getDefaultTargetTriple());
  if (tensorTarget.usesX86() &&
      processTriple.getArch() != llvm::Triple::x86_64) {
    llvm::errs()
        << "error: x86 tensor target cannot be JIT-executed from a non-x86_64 "
           "process; build/run lyc as x86_64 under Rosetta or use AOT/emit-llvm"
        << "\n";
    return failure();
  }
  if (tensorTarget.usesArmSME() && !processTriple.isAArch64()) {
    llvm::errs()
        << "error: arm-sme tensor target cannot be JIT-executed from a "
           "non-aarch64 process\n";
    return failure();
  }

  std::unique_ptr<llvm::orc::LLJIT> jit;
  llvm::orc::ExecutorAddr entryAddress;
  llvm::orc::ExecutorAddr runnerAddress;

  {
    PerfScope perf("jit-build");
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto tmBuilderOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrErr) {
      llvm::errs() << "Failed to create JITTargetMachineBuilder: "
                   << llvm::toString(tmBuilderOrErr.takeError()) << "\n";
      return failure();
    }
    auto tmBuilder = std::move(*tmBuilderOrErr);
    tmBuilder.setCPU(codeGenCPUNameForTarget(tensorTarget, processTriple));
    tmBuilder.setFeatures(
        codeGenFeaturesForTarget(tensorTarget, processTriple));
    auto options = tmBuilder.getOptions();
    options.ExceptionModel = exceptionModelForTargetTriple(processTriple);
    options.MCOptions.EmitCompactUnwindNonCanonical = true;
    options.ForceDwarfFrameSection = true;
    options.MCOptions.EmitDwarfUnwind = llvm::EmitDwarfUnwindType::Always;
    tmBuilder.setOptions(options);

    auto optimizationTMBuilder = tmBuilder;
    auto optimizationTargetMachineOrErr =
        optimizationTMBuilder.createTargetMachine();
    if (!optimizationTargetMachineOrErr) {
      llvm::errs() << "Failed to create optimization TargetMachine: "
                   << llvm::toString(optimizationTargetMachineOrErr.takeError())
                   << "\n";
      return failure();
    }
    std::unique_ptr<llvm::TargetMachine> optimizationTargetMachine =
        std::move(*optimizationTargetMachineOrErr);

    auto compileFunctionCreator = [](llvm::orc::JITTargetMachineBuilder jtmb)
        -> llvm::Expected<
            std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
      auto tmOrErr = jtmb.createTargetMachine();
      if (!tmOrErr)
        return tmOrErr.takeError();
      return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
          std::move(*tmOrErr));
    };

    auto objectLayerCreator =
        [](llvm::orc::ExecutionSession &session,
           const llvm::Triple &tt) -> std::unique_ptr<llvm::orc::ObjectLayer> {
      auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);
      if (tt.isOSBinFormatCOFF()) {
        layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
        layer->setAutoClaimResponsibilityForObjectSymbols(true);
      }
      return layer;
    };

    llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(tmBuilder))
        .setCompileFunctionCreator(compileFunctionCreator)
        .setObjectLinkingLayerCreator(objectLayerCreator)
        .setPrePlatformSetup([](llvm::orc::LLJIT &jit) -> llvm::Error {
          auto psjd = jit.getProcessSymbolsJITDylib();
          if (!psjd)
            return llvm::make_error<llvm::StringError>(
                "Native platforms require a process symbols JITDylib",
                llvm::inconvertibleErrorCode());
          auto dlGen =
              llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                  jit.getDataLayout().getGlobalPrefix());
          if (dlGen)
            psjd->addGenerator(std::move(*dlGen));
#if defined(__APPLE__)
          const char *libcppCandidates[] = {
              "/usr/lib/libc++.1.dylib",
              "libc++.1.dylib",
              "/usr/lib/libc++abi.dylib",
              "libc++abi.dylib",
          };
          for (const char *lib : libcppCandidates) {
            if (auto gen = llvm::orc::DynamicLibrarySearchGenerator::Load(
                    lib, jit.getDataLayout().getGlobalPrefix())) {
              psjd->addGenerator(std::move(*gen));
            } else {
              llvm::consumeError(gen.takeError());
            }
          }
#endif
          return llvm::Error::success();
        });
    auto jitExpected = builder.create();
    if (!jitExpected) {
      llvm::errs() << "Failed to create LLJIT\n";
      return failure();
    }
    jit = std::move(*jitExpected);

    LLVMSafetyProfile safetyProfile;
    collectLLVMSafetyContracts(module, safetyProfile);
    llvm::SmallVector<py::PythonCallSiteRange, 16> pythonCallSites;
    py::collectPythonCallSiteRanges(module, pythonCallSites);
    attachPythonDebugInfo(module);

    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
    if (!llvmModule) {
      llvm::errs() << "Failed to translate to LLVM IR\n";
      return failure();
    }
    py::installPythonExceptionCleanupFrames(*llvmModule, pythonCallSites);
    if (failed(verifyLLVMIRSafetyMetadataAttached(*llvmModule, safetyProfile)))
      return failure();
    if (failed(verifyPostCoroLLVMThreadSafety(*llvmModule, safetyProfile)))
      return failure();
    for (auto &func : *llvmModule) {
      if (!func.isDeclaration())
        func.setUWTableKind(llvm::UWTableKind::Async);
    }

    llvmModule->setDataLayout(jit->getDataLayout());
    llvmModule->setTargetTriple(jit->getTargetTriple().getTriple());
    if (failed(linkPrelinkedRuntime(*llvmModule)))
      return failure();
    if (failed(linkEmbeddedNativeRuntime(*llvmModule)))
      return failure();
    runLLVMCoroLowering(*llvmModule, optimizationTargetMachine.get());
    dumpLLVMForPass(irDump, "llvm-translation", *llvmModule);
    if (failed(verifyOptimizedLLVMThreadSafety(*llvmModule, safetyProfile)))
      return failure();

    auto &jd = jit->getMainJITDylib();
    auto dlGen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        jit->getDataLayout().getGlobalPrefix());
    if (dlGen)
      jd.addGenerator(std::move(*dlGen));
    llvm::orc::MangleAndInterner interner(jd.getExecutionSession(),
                                          jit->getDataLayout());
    auto symbolMap = buildRuntimeSymbolMap(interner);
    cantFail(jd.define(absoluteSymbols(std::move(symbolMap))));

    auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmModule),
                                           std::move(llvmContext));
    if (auto err = jit->addIRModule(std::move(tsm))) {
      llvm::errs() << "JIT add module error: " << err << "\n";
      return failure();
    }
    if (auto err = jit->initialize(jd)) {
      llvm::errs() << "JIT initialize error: " << err << "\n";
      return failure();
    }

    auto sym = jit->lookup("__main__");
    if (!sym) {
      llvm::errs() << "JIT lookup failed: " << sym.takeError() << "\n";
      return failure();
    }
    entryAddress = *sym;
    auto runnerSym = jit->lookup("LyRunPythonMain");
    if (!runnerSym) {
      llvm::errs() << "JIT runtime lookup failed: " << runnerSym.takeError()
                   << "\n";
      return failure();
    }
    runnerAddress = *runnerSym;
  }

  auto *entry = entryAddress.toPtr<void (*)()>();
  auto *runner = runnerAddress.toPtr<int (*)(void (*)())>();
  {
    PerfScope perf("execution");
    if (runner(entry) != 0)
      return failure();
  }
  return success();
}

} // namespace

static llvm::cl::OptionCategory LythonCategory("lython options");
static llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output file (default: a.out)"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("a.out"),
                   llvm::cl::cat(LythonCategory));
static llvm::cl::opt<bool> EmitLLVMOnly(
    "emit-llvm",
    llvm::cl::desc("Stop after emitting LLVM IR to the output file"),
    llvm::cl::init(false), llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string>
    TargetOption("target",
                 llvm::cl::desc("Generate code for the given target triple"),
                 llvm::cl::value_desc("triple"), llvm::cl::init(""),
                 llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string>
    TargetCPUOption("mcpu", llvm::cl::desc("Target a specific CPU name"),
                    llvm::cl::value_desc("cpu-name"), llvm::cl::init(""),
                    llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string> TargetFPUOption(
    "mfpu", llvm::cl::desc("Target a specific FPU name for the linker driver"),
    llvm::cl::value_desc("fpu-name"), llvm::cl::init(""),
    llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string>
    TargetFloatABIOption("mfloat-abi",
                         llvm::cl::desc("Target floating-point ABI"),
                         llvm::cl::value_desc("abi"), llvm::cl::init(""),
                         llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string> TargetSysrootOption(
    "sysroot",
    llvm::cl::desc("Use a target sysroot when linking AOT executables"),
    llvm::cl::value_desc("path"), llvm::cl::init(""),
    llvm::cl::cat(LythonCategory));
static llvm::cl::list<std::string>
    IncludePathOptions("I",
                       llvm::cl::desc("Add a clang-style include search path"),
                       llvm::cl::value_desc("path"), llvm::cl::Prefix,
                       llvm::cl::ZeroOrMore, llvm::cl::cat(LythonCategory));
static llvm::cl::list<std::string>
    LibraryPathOptions("L",
                       llvm::cl::desc("Add a clang-style library search path"),
                       llvm::cl::value_desc("path"), llvm::cl::Prefix,
                       llvm::cl::ZeroOrMore, llvm::cl::cat(LythonCategory));
static llvm::cl::SubCommand JitCommand("jit", "JIT execute an input file");
static llvm::cl::opt<std::string>
    JitInputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                     llvm::cl::Required, llvm::cl::sub(JitCommand));
static llvm::cl::SubCommand ParseCommand(
    "parse",
    "Parse Python source with the vendored CPython 3.14 PEG/ASDL C++ frontend");
static llvm::cl::opt<std::string>
    ParseInputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                       llvm::cl::Required, llvm::cl::sub(ParseCommand));
static llvm::cl::opt<bool> ParseTypeComments(
    "type-comments",
    llvm::cl::desc("Include CPython-style type comments in the dumped C++ AST"),
    llvm::cl::init(false), llvm::cl::sub(ParseCommand));
static llvm::cl::opt<bool> ParseAttributes(
    "attributes",
    llvm::cl::desc("Include CPython-style source location attributes in the "
                   "dumped C++ AST"),
    llvm::cl::init(false), llvm::cl::sub(ParseCommand));
static llvm::cl::opt<bool>
    ParseInteractiveMode("interactive",
                         llvm::cl::desc("Parse input as a single interactive "
                                        "statement, like ast.parse(..., "
                                        "mode=\"single\")"),
                         llvm::cl::init(false), llvm::cl::sub(ParseCommand));
static llvm::cl::opt<bool>
    ParseExpressionMode("expression",
                        llvm::cl::desc("Parse input as a single expression"),
                        llvm::cl::init(false), llvm::cl::sub(ParseCommand));
static llvm::cl::opt<bool> ParseFunctionTypeMode(
    "function-type",
    llvm::cl::desc("Parse input as a CPython function type comment"),
    llvm::cl::init(false), llvm::cl::sub(ParseCommand));

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &os) { os << "Lython CLI based on MLIR\n"; });
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Lython compiler driver (clang-style)\n");
  TargetTriple = TargetOption;
  TargetCPU = TargetCPUOption;
  TargetFPU = TargetFPUOption;
  TargetFloatABI = TargetFloatABIOption;
  TargetSysroot = TargetSysrootOption;
  IncludeSearchPaths.assign(IncludePathOptions.begin(),
                            IncludePathOptions.end());
  LibrarySearchPaths.assign(LibraryPathOptions.begin(),
                            LibraryPathOptions.end());
  IRDumpConfig irDump = IRDumpConfig::fromEnv();

  const bool jitMode = static_cast<bool>(JitCommand);
  const bool parseMode = static_cast<bool>(ParseCommand);
  std::string inputPath;
  std::string outputPath = OutputFilename;

  if (parseMode) {
    inputPath = ParseInputFilename;
  } else if (jitMode) {
    inputPath = JitInputFilename;
  } else {
    if (InputFilename.empty()) {
      llvm::errs() << "error: no input files\n";
      llvm::cl::PrintHelpMessage();
      return 1;
    }
    inputPath = InputFilename;
  }

  if (parseMode) {
    const int parseModeCount = (ParseInteractiveMode ? 1 : 0) +
                               (ParseExpressionMode ? 1 : 0) +
                               (ParseFunctionTypeMode ? 1 : 0);
    if (parseModeCount > 1) {
      llvm::errs() << "error: --interactive, --expression, and "
                      "--function-type are mutually exclusive\n";
      return 1;
    }

    return failed(runCppParserDump(inputPath, ParseTypeComments,
                                   ParseAttributes, ParseInteractiveMode,
                                   ParseExpressionMode, ParseFunctionTypeMode))
               ? 1
               : 0;
  }

  bool isPythonInput = llvm::sys::path::extension(inputPath) == ".py";

  DialectRegistry registry;
  registry.insert<py::PyDialect, affine::AffineDialect, async::AsyncDialect,
                  func::FuncDialect, arith::ArithDialect, scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, tensor::TensorDialect,
                  linalg::LinalgDialect, memref::MemRefDialect,
                  vector::VectorDialect, bufferization::BufferizationDialect,
                  LLVM::LLVMDialect>();
  py::runtime_lowering::arch::arm::registerSMEDialects(registry);
  py::runtime_lowering::arch::x86::registerX86Dialects(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  py::runtime_lowering::arch::x86::registerX86Translations(registry);
  registerPySafetyLLVMIRTranslation(registry);

  MLIRContext context;
  context.appendDialectRegistry(registry);

  std::string mlirSource;
  OwningOpRef<ModuleOp> module;

  if (isPythonInput) {
    if (failed(generateModuleFromCppFrontend(inputPath, context, module)))
      return 1;
  } else {
    auto file = mlir::openInputFile(inputPath);
    if (!file) {
      llvm::errs() << "error: could not open input file '" << inputPath
                   << "'\n";
      return 1;
    }
    mlirSource = std::string(file->getBuffer());
  }

  if (!module)
    module = parseModuleFromBuffer(mlirSource, context);
  if (!module)
    return 1;

  py::TensorLoweringTarget tensorTarget =
      detectTensorLoweringTarget(codeGenTripleForTarget({}));

  {
    PerfScope perf("lowering");
    if (failed(runPipeline(*module, context, irDump, tensorTarget))) {
      llvm::errs() << "Failed to run lowering pipeline\n";
      return 1;
    }
  }

  if (jitMode) {
    return failed(runJIT(*module, irDump, tensorTarget)) ? 1 : 0;
  }

  LLVMSafetyProfile safetyProfile;
  collectLLVMSafetyContracts(*module, safetyProfile);
  llvm::SmallVector<py::PythonCallSiteRange, 16> pythonCallSites;
  py::collectPythonCallSiteRanges(*module, pythonCallSites);
  attachPythonDebugInfo(*module);

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }
  py::installPythonExceptionCleanupFrames(*llvmModule, pythonCallSites);
  dumpLLVMForPass(irDump, "llvm-translation", *llvmModule);
  if (failed(verifyLLVMIRSafetyMetadataAttached(*llvmModule, safetyProfile)))
    return 1;
  if (failed(verifyPostCoroLLVMThreadSafety(*llvmModule, safetyProfile)))
    return 1;

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  if (failed(configureLLVMModuleCodeGenTarget(*llvmModule, tensorTarget)))
    return 1;

  if (py::runtime_library::prelowerGenerationMode()) {
    sealRuntimeSafetyMetadata(*llvmModule);
  } else {
    if (failed(linkPrelinkedRuntime(*llvmModule)))
      return 1;
    if (failed(linkEmbeddedNativeRuntime(*llvmModule)))
      return 1;
  }
  rewriteExceptionPersonalityForTarget(*llvmModule);

  if (!py::runtime_library::prelowerGenerationMode() &&
      failed(installAOTEntryPoint(*llvmModule)))
    return 1;

  if (EmitLLVMOnly)
    return failed(writeLLVMIR(*llvmModule, outputPath)) ? 1 : 0;

  return failed(buildExecutable(*llvmModule, safetyProfile, tensorTarget,
                                outputPath))
             ? 1
             : 0;
}
