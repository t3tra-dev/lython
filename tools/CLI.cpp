#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Error.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

#include "Common/Instrumentation.h"
#include "Common/LoweringPipeline.h"
#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "embedded.h"
#include "Emitter.h"
#include "Parser.h"
#include "Passes/Runtime/Arch/Arm/PrimitiveTensorArmSME.h"
#include "Passes/Runtime/Arch/X86/PrimitiveTensorX86.h"
#include "Driver.h"
#include "DriverCodeGen.h"
#include "SanitizerSupport.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;
using namespace lython::driver;

namespace {
using py::PerfScope;

#ifndef LYTHON_LLVM_TOOLS_BINARY_DIR
#define LYTHON_LLVM_TOOLS_BINARY_DIR ""
#endif

lython::driver::DriverOptions Options;
std::vector<std::string> IncludeSearchPaths;
std::vector<std::string> LibrarySearchPaths;

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

// Coro lowering, linked-contract collection, and the optimized thread-safety
// check are identical on every codegen exit (JIT, object file, --emit-llvm);
// splitting them per exit is how the safety checks drifted historically.
LogicalResult finalizeLoweredLLVMModule(llvm::Module &llvmModule,
                                        LLVMSafetyProfile &safetyProfile,
                                        llvm::TargetMachine *targetMachine,
                                        llvm::OptimizationLevel optLevel,
                                        const py::IRDumpConfig *irDump) {
  runLLVMCoroLowering(llvmModule, Options.sanitizers, targetMachine, optLevel);
  if (!Options.releaseMode)
    collectLinkedLLVMSafetyContracts(llvmModule, safetyProfile);
  if (irDump)
    dumpLLVMForPass(*irDump, "llvm-translation", llvmModule);
  if (!Options.releaseMode &&
      failed(verifyOptimizedLLVMThreadSafe(llvmModule, safetyProfile,
                                           llvm::errs())))
    return failure();
  return success();
}

LogicalResult emitObjectFile(llvm::Module &llvmModule,
                             LLVMSafetyProfile &safetyProfile,
                             py::TensorLoweringTarget tensorTarget,
                             StringRef objectPath) {
  std::string targetTriple;
  auto targetMachine = createCodeGenTargetMachine(tensorTarget, Options,
                                                  &targetTriple, llvm::errs());
  if (!targetMachine)
    return failure();
  llvmModule.setTargetTriple(llvm::Triple(targetTriple));
  llvmModule.setDataLayout(targetMachine->createDataLayout());
  if (failed(finalizeLoweredLLVMModule(llvmModule, safetyProfile,
                                       targetMachine.get(),
                                       llvm::OptimizationLevel::O2,
                                       /*irDump=*/nullptr)))
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

std::optional<std::string> findLLVMToolProgram(llvm::StringRef name) {
  llvm::StringRef toolsDir = LYTHON_LLVM_TOOLS_BINARY_DIR;
  if (!toolsDir.empty()) {
    llvm::SmallString<256> path(toolsDir);
    llvm::sys::path::append(path, name);
    if (llvm::sys::fs::can_execute(path))
      return path.str().str();
  }
  return findExecutableProgram(name);
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
  llvm::Triple targetTriple = codeGenTripleForTarget(tensorTarget, Options);
  if (targetTriple.isWindowsGNUEnvironment()) {
    if (auto mingw = findMinGWLinkerDriver(targetTriple))
      return LinkerDriver{*mingw, LinkerDriverFlavor::MinGWGcc};
  }

  auto clangExe = findLLVMToolProgram("clang++");
  if (!clangExe)
    clangExe = findLLVMToolProgram("clang");
  if (!clangExe) {
    llvm::errs() << "error: clang++/clang executable not found in PATH\n";
    return std::nullopt;
  }
  return LinkerDriver{*clangExe, LinkerDriverFlavor::Clang};
}

void appendLinkTargetArgs(std::vector<std::string> &args,
                          py::TensorLoweringTarget tensorTarget,
                          LinkerDriverFlavor driverFlavor) {
  llvm::Triple targetTriple = codeGenTripleForTarget(tensorTarget, Options);
  llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  std::string sysroot = configuredTargetSysrootOverride(Options);
  if (driverFlavor == LinkerDriverFlavor::Clang &&
      targetTriple.normalize() != hostTriple.normalize()) {
    args.emplace_back("-target");
    args.emplace_back(targetTriple.normalize());
  }
  if (!sysroot.empty())
    args.emplace_back("--sysroot=" + sysroot);
  if (driverFlavor == LinkerDriverFlavor::Clang && !Options.targetCPU.empty())
    args.emplace_back("-mcpu=" + Options.targetCPU);
  if (driverFlavor == LinkerDriverFlavor::Clang && !Options.targetFPU.empty())
    args.emplace_back("-mfpu=" + Options.targetFPU);
  if (driverFlavor == LinkerDriverFlavor::Clang &&
      !Options.targetFloatABI.empty())
    args.emplace_back("-mfloat-abi=" + Options.targetFloatABI);
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

void appendSanitizerLinkArgs(std::vector<std::string> &args,
                             const SanitizerConfig &sanitizers) {
  std::string sanitizerList = lython::driver::sanitizerClangList(sanitizers);
  if (sanitizerList.empty())
    return;
  args.emplace_back("-fsanitize=" + sanitizerList);
}

void appendLinkTargetLibraries(std::vector<std::string> &args,
                               py::TensorLoweringTarget tensorTarget) {
  llvm::Triple targetTriple = codeGenTripleForTarget(tensorTarget, Options);
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
  int result = llvm::sys::ExecuteAndWait(clangProgram, args, std::nullopt, {},
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
  if (Options.sanitizers.any() &&
      linker->flavor != LinkerDriverFlavor::Clang) {
    llvm::errs() << "error: -fsanitize is only supported with the clang linker "
                    "driver\n";
    return failure();
  }

  std::vector<std::string> argStorage;
  argStorage.push_back(linker->program);
  appendLinkTargetArgs(argStorage, tensorTarget, linker->flavor);
  appendSanitizerLinkArgs(argStorage, Options.sanitizers);
  argStorage.emplace_back(objectPath.str());
  appendLinkTargetLibraries(argStorage, tensorTarget);
  argStorage.emplace_back("-O2");
  if (codeGenTripleForTarget(tensorTarget, Options).isOSLinux())
    argStorage.emplace_back("-no-pie");
  argStorage.emplace_back("-o");
  argStorage.emplace_back(outputPath.str());

  return runLinkerCommand(linker->program, argStorage, "error: linking failed");
}

LogicalResult buildExecutable(llvm::Module &llvmModule,
                              LLVMSafetyProfile &safetyProfile,
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

static llvm::orc::shared::CWrapperFunctionBuffer
noopDeregisterEHFrameSectionAllocAction(const char *argData, size_t argSize) {
  return llvm::orc::shared::WrapperFunction<llvm::orc::shared::SPSError(
      llvm::orc::shared::SPSExecutorAddrRange)>::
      handle(argData, argSize,
             [](llvm::orc::ExecutorAddrRange) {
               return llvm::Error::success();
             })
          .release();
}

FailureOr<int> runJIT(ModuleOp module, const py::IRDumpConfig &irDump,
                      py::TensorLoweringTarget tensorTarget,
                      llvm::ArrayRef<std::string> programArgs) {
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
  llvm::orc::ExecutorAddr initArgsAddress;

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
    tmBuilder.setCPU(
        codeGenCPUNameForTarget(tensorTarget, processTriple, Options));
    tmBuilder.setFeatures(
        codeGenFeaturesForTarget(tensorTarget, processTriple, Options));
    // JIT favors first-output latency; AOT still uses the optimized pipeline.
    tmBuilder.setCodeGenOptLevel(llvm::CodeGenOptLevel::None);
    auto options = tmBuilder.getOptions();
    lython::driver::applyExceptionUnwindOptions(options, processTriple);
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

    auto objectLayerCreator = [](llvm::orc::ExecutionSession &session)
        -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
      auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);
      const llvm::Triple &tt = session.getTargetTriple();
      if (tt.isOSBinFormatELF()) {
        // GitHub Actions' Linux runner can abort in libgcc's
        // __deregister_frame while tearing LLJIT down after successful program
        // execution. Keep EH frame registration for Python exception unwinding,
        // but skip deregistration for this single-shot CLI JIT lifetime.
        layer->addPlugin(std::make_shared<llvm::orc::EHFrameRegistrationPlugin>(
            llvm::orc::ExecutorAddr::fromPtr(
                &llvm_orc_registerEHFrameSectionAllocAction),
            llvm::orc::ExecutorAddr::fromPtr(
                &noopDeregisterEHFrameSectionAllocAction)));
      }
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
    {
      PerfScope perf("jit-build.create-lljit");
      auto jitExpected = builder.create();
      if (!jitExpected) {
        llvm::errs() << "Failed to create LLJIT\n";
        return failure();
      }
      jit = std::move(*jitExpected);
      if (failed(lython::driver::addJITSanitizerRuntimes(
              *jit, Options.sanitizers, processTriple)))
        return failure();
    }

    lython::driver::VerifiedLLVMModule verified;
    {
      PerfScope perf("jit-build.translate-to-llvm");
      if (failed(lython::driver::translateToVerifiedLLVMIR(
              module, Options, irDump, verified, llvm::errs())))
        return failure();
    }
    std::unique_ptr<llvm::LLVMContext> llvmContext =
        std::move(verified.llvmContext);
    std::unique_ptr<llvm::Module> llvmModule = std::move(verified.llvmModule);
    LLVMSafetyProfile safetyProfile = std::move(verified.safetyProfile);
    for (auto &func : *llvmModule) {
      if (!func.isDeclaration())
        func.setUWTableKind(llvm::UWTableKind::Async);
    }

    llvmModule->setDataLayout(jit->getDataLayout());
    llvmModule->setTargetTriple(jit->getTargetTriple());
    {
      PerfScope perf("jit-build.link-runtime");
      if (failed(py::runtime_library::linkEmbeddedNativeRuntime(*llvmModule)))
        return failure();
      if (!Options.releaseMode)
        collectLinkedLLVMSafetyContracts(*llvmModule, safetyProfile);
    }
    {
      PerfScope perf("jit-build.llvm-opt");
      if (failed(finalizeLoweredLLVMModule(*llvmModule, safetyProfile,
                                           optimizationTargetMachine.get(),
                                           llvm::OptimizationLevel::O0,
                                           &irDump)))
        return failure();
    }

    auto &jd = jit->getMainJITDylib();
    {
      PerfScope perf("jit-build.define-runtime-symbols");
      auto dlGen =
          llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
              jit->getDataLayout().getGlobalPrefix());
      if (dlGen)
        jd.addGenerator(std::move(*dlGen));
    }

    auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmModule),
                                           std::move(llvmContext));
    {
      PerfScope perf("jit-build.add-ir-module");
      if (auto err = jit->addIRModule(std::move(tsm))) {
        llvm::errs() << "JIT add module error: " << err << "\n";
        return failure();
      }
    }
    {
      PerfScope perf("jit-build.initialize");
      if (auto err = jit->initialize(jd)) {
        llvm::errs() << "JIT initialize error: " << err << "\n";
        return failure();
      }
    }

    {
      PerfScope perf("jit-build.lookup-entrypoints");
      {
        PerfScope perf("jit-build.lookup-main");
        auto sym = jit->lookup("__main__");
        if (!sym) {
          llvm::errs() << "JIT lookup failed: " << sym.takeError() << "\n";
          return failure();
        }
        entryAddress = *sym;
      }
      {
        PerfScope perf("jit-build.lookup-runner");
        auto runnerSym = jit->lookup("LyRunPythonMain");
        if (!runnerSym) {
          llvm::errs() << "JIT runtime lookup failed: " << runnerSym.takeError()
                       << "\n";
          return failure();
        }
        runnerAddress = *runnerSym;
      }
      {
        PerfScope perf("jit-build.lookup-init-args");
        auto initArgsSym = jit->lookup("LyHost_InitArgs");
        if (!initArgsSym) {
          llvm::errs() << "JIT runtime lookup failed: "
                       << initArgsSym.takeError() << "\n";
          return failure();
        }
        initArgsAddress = *initArgsSym;
      }
    }
  }

  auto *entry = entryAddress.toPtr<void (*)()>();
  auto *runner = runnerAddress.toPtr<int (*)(void (*)())>();
  auto *initArgs = initArgsAddress.toPtr<void (*)(int, char **)>();
  // sys.argv is [script path, program args...] like CPython's; the storage
  // must outlive the run because LySys_GetArgv reads the raw vector lazily.
  std::vector<std::string> argvStorage(programArgs.begin(), programArgs.end());
  std::vector<char *> argvPointers;
  argvPointers.reserve(argvStorage.size());
  for (std::string &arg : argvStorage)
    argvPointers.push_back(arg.data());
  initArgs(static_cast<int>(argvPointers.size()), argvPointers.data());
  int status = 0;
  {
    PerfScope perf("execution");
    if (Options.sanitizers.leak)
      lython::driver::callLeakSanitizerHook("__lsan_enable");
    status = runner(entry);
    if (Options.sanitizers.leak)
      lython::driver::callLeakSanitizerHook("__lsan_disable");
  }
  return status;
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
static llvm::cl::opt<bool> AuditRuntimeManifest(
    "audit-runtime-manifest",
    llvm::cl::desc("Verify manifest runtime-required contracts against "
                   "runtime ABI symbols after runtime import"),
    llvm::cl::init(false), llvm::cl::cat(LythonCategory));
static llvm::cl::opt<bool> ReleaseModeOption(
    "release",
    llvm::cl::desc("Disable compiler verifier passes and hide user paths in "
                   "Python debug locations"),
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
static llvm::cl::list<std::string> SanitizerEnableOptions(
    "fsanitize",
    llvm::cl::desc("Enable runtime checks for address, leak, thread, or "
                   "undefined behavior"),
    llvm::cl::value_desc("checks"), llvm::cl::CommaSeparated,
    llvm::cl::ZeroOrMore, llvm::cl::callback([](const std::string &value) {
      lython::driver::recordSanitizerAction(/*enable=*/true, value);
    }),
    llvm::cl::cat(LythonCategory));
static llvm::cl::list<std::string> SanitizerDisableOptions(
    "fno-sanitize",
    llvm::cl::desc("Disable previously enabled sanitizer checks"),
    llvm::cl::value_desc("checks"), llvm::cl::CommaSeparated,
    llvm::cl::ZeroOrMore, llvm::cl::callback([](const std::string &value) {
      lython::driver::recordSanitizerAction(/*enable=*/false, value);
    }),
    llvm::cl::cat(LythonCategory));
static llvm::cl::SubCommand JitCommand("jit", "JIT execute an input file");
static llvm::cl::opt<std::string>
    JitInputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                     llvm::cl::Required, llvm::cl::sub(JitCommand));
static llvm::cl::list<std::string>
    JitProgramArgs(llvm::cl::ConsumeAfter,
                   llvm::cl::desc("<program arguments>..."),
                   llvm::cl::sub(JitCommand));
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
  // Pre-lowered runtime-internal lib modules (generated
  // embedded_lib_internal.cpp) join the native runtime link set.
  py::runtime_library::embedded::registerPyRuntimeEmbeddedModules();
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &os) { os << "Lython CLI based on MLIR\n"; });
  bool releaseModeFromArgv = false;
  std::vector<const char *> filteredArgv;
  filteredArgv.reserve(argc);
  filteredArgv.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    if (arg == "--release" || arg == "-release") {
      releaseModeFromArgv = true;
      continue;
    }
    filteredArgv.push_back(argv[index]);
  }

  llvm::cl::ParseCommandLineOptions(static_cast<int>(filteredArgv.size()),
                                    filteredArgv.data(),
                                    "Lython compiler driver (clang-style)\n");
  Options.targetTriple = TargetOption;
  Options.targetCPU = TargetCPUOption;
  Options.targetFPU = TargetFPUOption;
  Options.targetFloatABI = TargetFloatABIOption;
  Options.targetSysroot = TargetSysrootOption;
  IncludeSearchPaths.assign(IncludePathOptions.begin(),
                            IncludePathOptions.end());
  LibrarySearchPaths.assign(LibraryPathOptions.begin(),
                            LibraryPathOptions.end());
  Options.releaseMode = releaseModeFromArgv || ReleaseModeOption;
  Options.auditRuntimeManifest = AuditRuntimeManifest;
  py::IRDumpConfig irDump = py::IRDumpConfig::fromEnv();

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

  if (failed(lython::driver::buildSanitizerConfig(Options.sanitizers)))
    return 1;
  if (jitMode && Options.sanitizers.any()) {
    llvm::Triple processTriple(llvm::sys::getDefaultTargetTriple());
    if (failed(lython::driver::ensureJITSanitizerRuntimesPreloaded(
            argv, Options.sanitizers, processTriple)))
      return 1;
  }
  if (jitMode && Options.sanitizers.leak)
    lython::driver::callLeakSanitizerHook("__lsan_disable");

  bool isPythonInput = llvm::sys::path::extension(inputPath) == ".py";

  DialectRegistry registry;
  lython::driver::registerLythonDialects(registry);

  MLIRContext context;
  context.appendDialectRegistry(registry);

  std::string mlirSource;
  OwningOpRef<ModuleOp> module;

  if (isPythonInput) {
    if (failed(emitMLIRFromFile(inputPath, Options, context, module,
                                llvm::errs())))
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
    module = parseModuleFromBuffer(mlirSource, context, llvm::errs());
  if (!module)
    return 1;

  py::TensorLoweringTarget tensorTarget = detectTensorLoweringTarget(Options);
  if (failed(stampTargetPlatformFacts(*module, tensorTarget, Options,
                                      llvm::errs())))
    return 1;

  {
    PerfScope perf("lowering");
    py::LoweringPipelineOptions loweringOptions;
    loweringOptions.auditRuntimeManifest = Options.auditRuntimeManifest;
    loweringOptions.enableVerifiers = !Options.releaseMode;
    if (failed(py::runLoweringPipeline(*module, tensorTarget, irDump,
                                       loweringOptions))) {
      llvm::errs() << "Failed to run lowering pipeline\n";
      return 1;
    }
  }

  if (jitMode) {
    std::vector<std::string> programArgs;
    programArgs.push_back(JitInputFilename);
    programArgs.insert(programArgs.end(), JitProgramArgs.begin(),
                       JitProgramArgs.end());
    FailureOr<int> status = runJIT(*module, irDump, tensorTarget, programArgs);
    return failed(status) ? 1 : *status;
  }

  lython::driver::VerifiedLLVMModule verified;
  if (failed(translateToVerifiedLLVMIR(*module, Options, irDump, verified,
                                       llvm::errs())))
    return 1;
  llvm::Module &llvmModule = *verified.llvmModule;
  LLVMSafetyProfile &safetyProfile = verified.safetyProfile;

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  if (failed(configureLLVMModuleCodeGenTarget(llvmModule, tensorTarget,
                                              Options, llvm::errs())))
    return 1;

  if (failed(py::runtime_library::linkEmbeddedNativeRuntime(llvmModule)))
    return 1;
  rewriteExceptionPersonalityForTarget(llvmModule);

  if (failed(installAOTEntryPoint(llvmModule, llvm::errs())))
    return 1;

  if (!Options.releaseMode)
    collectLinkedLLVMSafetyContracts(llvmModule, safetyProfile);

  if (EmitLLVMOnly) {
    if (Options.sanitizers.requiresLLVMInstrumentation()) {
      std::string targetTriple;
      auto targetMachine = createCodeGenTargetMachine(
          tensorTarget, Options, &targetTriple, llvm::errs());
      if (!targetMachine)
        return 1;
      llvmModule.setTargetTriple(llvm::Triple(targetTriple));
      llvmModule.setDataLayout(targetMachine->createDataLayout());
      if (failed(finalizeLoweredLLVMModule(llvmModule, safetyProfile,
                                           targetMachine.get(),
                                           llvm::OptimizationLevel::O2,
                                           /*irDump=*/nullptr)))
        return 1;
    }
    return failed(writeLLVMIR(llvmModule, outputPath, llvm::errs())) ? 1 : 0;
  }

  return failed(buildExecutable(llvmModule, safetyProfile, tensorTarget,
                                outputPath))
             ? 1
             : 0;
}
