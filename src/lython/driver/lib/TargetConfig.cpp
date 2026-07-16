#include "Driver.h"
#include "DriverCodeGen.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/X86TargetParser.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

using namespace mlir;

namespace lython::driver {

static std::string trimEnvToken(llvm::StringRef token) {
  token = token.trim();
  return token.str();
}

static std::string
configuredTargetTripleOverride(const DriverOptions &options) {
  return trimEnvToken(options.targetTriple);
}

std::string configuredTargetSysrootOverride(const DriverOptions &options) {
  return trimEnvToken(options.targetSysroot);
}

static bool parseConfiguredFloatABI(llvm::FloatABI::ABIType &result,
                                    const DriverOptions &options,
                                    llvm::raw_ostream &diag) {
  llvm::StringRef value = options.targetFloatABI;
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
  diag << "error: unsupported -mfloat-abi value '" << value
       << "'; expected default, soft, softfp, or hard\n";
  return false;
}

static std::string hostCPUNameForCodeGen() {
  llvm::StringRef cpu = llvm::sys::getHostCPUName();
  if (cpu.empty())
    return "generic";
  return cpu.str();
}

static std::string hostCPUFeaturesForCodeGen() {
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

// A Mach-O object carries its platform in LC_BUILD_VERSION, which the MachO
// writer only emits when the triple names an OS version. `-target
// x86_64-apple-darwin` names none, so the object comes out without the load
// command and the linker falls back to guessing the platform ("assuming:
// macOS"). Borrow the version from the host, which is the only place a
// correct one exists when cross-compiling between Darwin architectures.
static void fillDarwinOSVersionFromHost(llvm::Triple &triple) {
  if (!triple.isOSDarwin() || !triple.getOSVersion().empty())
    return;
  llvm::Triple host(llvm::sys::getDefaultTargetTriple());
  if (!host.isOSDarwin() || host.getOSVersion().empty())
    return;
  triple.setOSName(
      (triple.getOSName() + host.getOSVersion().getAsString()).str());
}

llvm::Triple codeGenTripleForTarget(py::TensorLoweringTarget target,
                                    const DriverOptions &options) {
  (void)target;
  std::string override = configuredTargetTripleOverride(options);
  if (override.empty())
    return llvm::Triple(llvm::sys::getDefaultTargetTriple());
  llvm::Triple triple(override);
  fillDarwinOSVersionFromHost(triple);
  return triple;
}

static bool targetFeatureEnabled(const llvm::Triple &triple,
                                 llvm::StringRef feature,
                                 const DriverOptions &options);

static bool targetFeatureStringContains(llvm::StringRef features,
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

static void appendTargetFeature(std::string &features,
                                llvm::StringRef feature) {
  if (targetFeatureStringContains(features, feature))
    return;
  if (!features.empty())
    features += ",";
  features += "+";
  features += feature;
}

static std::string configuredCPUNameForCodeGen(const llvm::Triple &triple,
                                               const DriverOptions &options) {
  llvm::StringRef cpu = options.targetCPU;
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
                                    const llvm::Triple &triple,
                                    const DriverOptions &options) {
  std::string configuredCPU = configuredCPUNameForCodeGen(triple, options);
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
  return "";
}

std::string codeGenFeaturesForTarget(py::TensorLoweringTarget target,
                                     const llvm::Triple &triple,
                                     const DriverOptions &options) {
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
    if (targetFeatureEnabled(triple, "sme2", options))
      appendTargetFeature(features, "sme2");
    if (targetFeatureEnabled(triple, "sme-f64f64", options))
      appendTargetFeature(features, "sme-f64f64");
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

void applyExceptionUnwindOptions(llvm::TargetOptions &options,
                                 const llvm::Triple &triple) {
  options.ExceptionModel = exceptionModelForTargetTriple(triple);
  options.MCOptions.EmitCompactUnwindNonCanonical = true;
  options.ForceDwarfFrameSection = true;
  options.MCOptions.EmitDwarfUnwind = llvm::EmitDwarfUnwindType::Always;
}

std::unique_ptr<llvm::TargetMachine>
createCodeGenTargetMachine(py::TensorLoweringTarget target,
                           const DriverOptions &options,
                           std::string *normalizedTriple,
                           llvm::raw_ostream &diag) {
  llvm::Triple triple(codeGenTripleForTarget(target, options).normalize());
  std::string targetTripleName = triple.normalize();
  if (normalizedTriple)
    *normalizedTriple = targetTripleName;

  std::string error;
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!llvmTarget) {
    diag << "Failed to lookup target: " << error << "\n";
    return nullptr;
  }

  llvm::TargetOptions opt;
  applyExceptionUnwindOptions(opt, triple);
  if (!parseConfiguredFloatABI(opt.FloatABIType, options, diag))
    return nullptr;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      llvmTarget->createTargetMachine(
          triple, codeGenCPUNameForTarget(target, triple, options),
          codeGenFeaturesForTarget(target, triple, options), opt,
          std::nullopt));
  if (!targetMachine)
    diag << "Failed to create target machine for " << targetTripleName << "\n";
  return targetMachine;
}

static bool hostFeatureEnabled(llvm::StringRef feature) {
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
  else if (feature == "sme-f64f64")
    sysctlName = "hw.optional.arm.FEAT_SME_F64F64";
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

static bool isHostCodeGenTriple(const llvm::Triple &triple) {
  return triple.normalize() ==
         llvm::Triple(llvm::sys::getDefaultTargetTriple()).normalize();
}

static bool featureNameMatches(llvm::StringRef value,
                               llvm::StringRef feature) {
  value.consume_front("+");
  return value == feature;
}

static bool x86CPUFeatureEnabled(llvm::StringRef cpu,
                                 llvm::StringRef feature) {
  llvm::SmallVector<llvm::StringRef, 64> features;
  llvm::X86::getFeaturesForCPU(cpu, features, /*NeedPlus=*/false);
  return llvm::is_contained(features, feature);
}

static bool aarch64CPUFeatureEnabled(llvm::StringRef cpu,
                                     llvm::StringRef feature) {
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

static bool targetFeatureEnabled(const llvm::Triple &triple,
                                 llvm::StringRef feature,
                                 const DriverOptions &options) {
  llvm::StringRef cpu = options.targetCPU;
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

// AMX has no capability bit and no target feature: it is a property of the
// Apple Silicon SoC, resolved at run time. Compile-time all we can ask is
// whether the binary could ever meet one.
static bool couldTargetAppleAMX(const llvm::Triple &triple) {
  return triple.isAArch64() && triple.isOSDarwin() &&
         !triple.isSimulatorEnvironment() &&
         !triple.isMacCatalystEnvironment();
}

py::TensorLoweringTarget
detectTensorLoweringTarget(const DriverOptions &options) {
  llvm::Triple triple = codeGenTripleForTarget({}, options);
  py::TensorLoweringTarget target;

  bool smeAvailable =
      triple.isAArch64() && (targetFeatureEnabled(triple, "sme", options) ||
                             targetFeatureEnabled(triple, "sme2", options));

  // Prefer the documented ISA. Apple's AMX only fills the gap on Apple Silicon
  // without SME (M1-M3), where the alternative is plain NEON.
  if (smeAvailable) {
    target.architecture = py::TensorLoweringArchitecture::ArmSME;
    target.armSMEF64F64 = targetFeatureEnabled(triple, "sme-f64f64", options);
    target.armSME2 = targetFeatureEnabled(triple, "sme2", options);
  } else if (couldTargetAppleAMX(triple)) {
    target.architecture = py::TensorLoweringArchitecture::AppleAMX;
  }
  if (triple.getArch() == llvm::Triple::x86_64) {
    if (targetFeatureEnabled(triple, "avx2", options) &&
        targetFeatureEnabled(triple, "fma", options))
      target.architecture = py::TensorLoweringArchitecture::X86AVX2FMA;
    else if (targetFeatureEnabled(triple, "sse4.2", options))
      target.architecture = py::TensorLoweringArchitecture::X86SSE42;
  }
  return target;
}

static std::uint64_t pointerWidthForTarget(const llvm::Triple &triple) {
  if (triple.isArch64Bit())
    return 64;
  if (triple.isArch32Bit())
    return 32;
  return 0;
}

static std::uint64_t cLongWidthForTarget(const llvm::Triple &triple,
                                         std::uint64_t pointerWidth) {
  if (triple.isOSWindows())
    return 32;
  return pointerWidth == 64 ? 64 : 32;
}

LogicalResult stampTargetPlatformFacts(ModuleOp module,
                                       py::TensorLoweringTarget tensorTarget,
                                       const DriverOptions &options,
                                       llvm::raw_ostream &diag) {
  llvm::Triple triple = codeGenTripleForTarget(tensorTarget, options);
  std::uint64_t pointerWidth = pointerWidthForTarget(triple);
  if (pointerWidth == 0) {
    diag << "error: cannot derive pointer width for target triple '"
         << triple.normalize() << "'\n";
    return failure();
  }

  Builder builder(module.getContext());
  module->setAttr("ly.target.triple",
                  builder.getStringAttr(triple.normalize()));
  module->setAttr("ly.target.pointer_width",
                  builder.getI64IntegerAttr(pointerWidth));
  module->setAttr(
      "ly.target.c_long_width",
      builder.getI64IntegerAttr(cLongWidthForTarget(triple, pointerWidth)));
  return success();
}
LogicalResult
configureLLVMModuleCodeGenTarget(llvm::Module &llvmModule,
                                 py::TensorLoweringTarget tensorTarget,
                                 const DriverOptions &options,
                                 llvm::raw_ostream &diag) {
  std::string targetTriple;
  auto targetMachine =
      createCodeGenTargetMachine(tensorTarget, options, &targetTriple, diag);
  if (!targetMachine)
    return failure();

  llvmModule.setTargetTriple(llvm::Triple(targetTriple));
  llvmModule.setDataLayout(targetMachine->createDataLayout());
  return success();
}

} // namespace lython::driver
