#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "lyrt.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

// Forward declare our custom lowering pass factories
namespace py {
  std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createRuntimeLoweringPass();
  std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createRefCountInsertionPass();
  std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createNativeVerificationPass();
}

namespace {

  constexpr llvm::StringLiteral kPythonFrontendScript = R"PY(
import ast
import sys
from lython.mlir import ir
from lython.visitors._base import BaseVisitor

def main():
    if len(sys.argv) < 2:
        raise SystemExit("expected input.py")
    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as source_file:
        source = source_file.read()
    tree = ast.parse(source, filename=input_path)
    ctx = ir.Context()
    parser = BaseVisitor(ctx)
    parser.visit(tree)
    module = parser.module
    if not module.operation.verify():
        raise SystemExit("Generated module failed verification")
    sys.stdout.write(str(module))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
)PY";

  void setEnvVar(llvm::StringRef name, llvm::StringRef value) {
#if defined(_WIN32)
    _putenv_s(name.str().c_str(), value.str().c_str());
#else
    setenv(name.str().c_str(), value.str().c_str(), 1);
#endif
  }

  void unsetEnvVar(llvm::StringRef name) {
#if defined(_WIN32)
    _putenv_s(name.str().c_str(), "");
#else
    unsetenv(name.str().c_str());
#endif
  }

  LogicalResult runPipeline(ModuleOp module, MLIRContext& context) {
    PassManager pm(&context);
    // Verify @native functions before any transformations
    // This enforces the modal logic separation (Primitive World vs Object World)
    pm.addPass(py::createNativeVerificationPass());
    // Insert reference counting operations using Affine SSA (Linear Type) logic
    pm.addPass(py::createRefCountInsertionPass());
    // Early canonicalization and CSE for arith/func ops
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(py::createRuntimeLoweringPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    return pm.run(module);
  }

  void registerRuntimeSymbols(ExecutionEngine& engine) {
    engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
      llvm::orc::SymbolMap symbolMap;
      auto add = [&](llvm::StringRef name, auto* ptr) {
        symbolMap[interner(name)] = {
            llvm::orc::ExecutorAddr::fromPtr(ptr),
            llvm::JITSymbolFlags::Exported };
        };
      add("Ly_IncRef", &Ly_IncRef);
      add("Ly_DecRef", &Ly_DecRef);
      add("LyUnicode_FromUTF8", &LyUnicode_FromUTF8);
      add("LyTuple_New", &LyTuple_New);
      add("LyTuple_SetItem", &LyTuple_SetItem);
      add("Ly_GetEmptyTuple", &Ly_GetEmptyTuple);
      add("LyDict_New", &LyDict_New);
      add("LyDict_Insert", &LyDict_Insert);
      add("LyDict_GetItem", &LyDict_GetItem);
      add("Ly_GetNone", &Ly_GetNone);
      add("Ly_GetBuiltinPrint", &Ly_GetBuiltinPrint);
      add("Ly_CallVectorcall", &Ly_CallVectorcall);
      add("Ly_Call", &Ly_Call);
      add("LyLong_FromI64", &LyLong_FromI64);
      add("LyLong_FromString", &LyLong_FromString);
      add("LyLong_Add", &LyLong_Add);
      add("LyLong_Sub", &LyLong_Sub);
      add("LyLong_Compare", &LyLong_Compare);
      add("LyFloat_FromDouble", &LyFloat_FromDouble);
      add("LyFloat_Add", &LyFloat_Add);
      add("LyFloat_Sub", &LyFloat_Sub);
      add("LyBool_FromBool", &LyBool_FromBool);
      add("LyObject_Repr", &LyObject_Repr);
      add("LyNumber_Add", &LyNumber_Add);
      add("LyNumber_Sub", &LyNumber_Sub);
      add("LyNumber_Le", &LyNumber_Le);
      add("LyBool_AsBool", &LyBool_AsBool);
      return symbolMap;
      });
  }

  OwningOpRef<ModuleOp> parseModuleFromBuffer(StringRef buffer,
    MLIRContext& context) {
    auto module = parseSourceString<ModuleOp>(buffer, &context);
    if (!module)
      llvm::errs() << "error: failed to parse MLIR source\n";
    return module;
  }

  std::optional<std::string> findBuildRoot(StringRef argv0) {
    llvm::SmallString<256> exePath(argv0);
    if (auto ec = llvm::sys::fs::real_path(exePath, exePath)) {
      llvm::errs() << "error: unable to resolve executable path: " << ec.message()
        << "\n";
      return std::nullopt;
    }
    llvm::sys::path::remove_filename(exePath);

    llvm::SmallString<256> current(exePath);
    while (!current.empty()) {
      llvm::SmallString<256> cachePath(current);
      llvm::sys::path::append(cachePath, "CMakeCache.txt");
      if (llvm::sys::fs::exists(cachePath))
        return std::string(current.str());
      if (!llvm::sys::path::has_parent_path(current))
        break;
      llvm::sys::path::remove_filename(current);
    }
    return std::nullopt;
  }

  std::optional<std::string> findSourceRoot(StringRef buildRoot) {
    llvm::SmallString<256> cachePath(buildRoot);
    llvm::sys::path::append(cachePath, "CMakeCache.txt");
    auto bufferOrErr = llvm::MemoryBuffer::getFile(cachePath);
    if (!bufferOrErr)
      return std::nullopt;

    llvm::StringRef content = bufferOrErr->get()->getBuffer();
    llvm::StringRef key = "CMAKE_HOME_DIRECTORY:";
    size_t pos = content.find(key);
    if (pos == llvm::StringRef::npos)
      return std::nullopt;
    llvm::StringRef line = content.substr(pos);
    line = line.substr(0, line.find('\n'));
    size_t eq = line.find('=');
    if (eq == llvm::StringRef::npos)
      return std::nullopt;
    llvm::StringRef value = line.substr(eq + 1).trim();
    if (value.empty())
      return std::nullopt;
    return value.str();
  }

  LogicalResult generateMlirFromPython(StringRef pythonFile, StringRef sourceRoot,
    std::string& mlirBuffer) {
    auto findProjectPython = [&](llvm::StringRef executable) -> std::optional<std::string> {
      llvm::SmallString<256> candidate(sourceRoot);
      llvm::sys::path::append(candidate, ".venv");
#if defined(_WIN32)
      llvm::sys::path::append(candidate, "Scripts");
#else
      llvm::sys::path::append(candidate, "bin");
#endif
      llvm::SmallString<256> binary(candidate);
      llvm::sys::path::append(binary, executable);
      if (llvm::sys::fs::exists(binary))
        return std::string(binary.str());
      return std::nullopt;
      };

    std::optional<std::string> pythonExe;
#if defined(_WIN32)
    pythonExe = findProjectPython("python.exe");
#else
    pythonExe = findProjectPython("python3");
    if (!pythonExe)
      pythonExe = findProjectPython("python");
#endif
    if (!pythonExe) {
      auto sysPython = llvm::sys::findProgramByName("python3");
      if (!sysPython)
        sysPython = llvm::sys::findProgramByName("python");
      if (!sysPython) {
        llvm::errs() << "error: could not find python3/python executable\n";
        return failure();
      }
      pythonExe = *sysPython;
    }

    llvm::SmallString<256> scriptPath;
    if (auto ec =
      llvm::sys::fs::createTemporaryFile("lython_frontend", "py", scriptPath)) {
      llvm::errs() << "error: failed to create temporary frontend script: "
        << ec.message() << "\n";
      return failure();
    }
    llvm::FileRemover scriptCleanup(scriptPath);
    {
      std::error_code ec;
      llvm::raw_fd_ostream scriptFile(scriptPath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::errs() << "error: failed to open temporary frontend script: "
          << ec.message() << "\n";
        return failure();
      }
      scriptFile << kPythonFrontendScript.str();
    }

    auto existingPyPath = llvm::sys::Process::GetEnv("PYTHONPATH");
    llvm::SmallString<256> newPyPath(sourceRoot);
    llvm::sys::path::append(newPyPath, "src");
    if (existingPyPath && !existingPyPath->empty()) {
      newPyPath += llvm::sys::path::get_separator();
      newPyPath += *existingPyPath;
    }
    setEnvVar("PYTHONPATH", newPyPath);

    std::string command;
    {
      llvm::raw_string_ostream os(command);
      llvm::sys::printArg(os, *pythonExe, /*Quote=*/true);
      os << ' ';
      llvm::sys::printArg(os, scriptPath.str(), /*Quote=*/true);
      os << ' ';
      llvm::sys::printArg(os, pythonFile, /*Quote=*/true);
    }

#if defined(_WIN32)
    FILE* pipe = _popen(command.c_str(), "rb");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) {
      llvm::errs() << "error: failed to invoke python frontend\n";
      if (existingPyPath)
        setEnvVar("PYTHONPATH", *existingPyPath);
      else
        unsetEnvVar("PYTHONPATH");
      return failure();
    }

    mlirBuffer.clear();
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe))
      mlirBuffer.append(buffer);

#if defined(_WIN32)
    int result = _pclose(pipe);
#else
    int result = pclose(pipe);
#endif

    if (existingPyPath)
      setEnvVar("PYTHONPATH", *existingPyPath);
    else
      unsetEnvVar("PYTHONPATH");

    if (result != 0) {
      llvm::errs() << "error: python frontend failed for '" << pythonFile << "'\n";
      return failure();
    }
    return success();
  }

  LogicalResult dumpLLVMIR(llvm::Module& llvmModule, StringRef outputPath) {
    std::error_code ec;
    llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
      return failure();
    }
    llvmModule.print(out, nullptr);
    return success();
  }

  LogicalResult emitObjectFile(llvm::Module& llvmModule, StringRef objectPath) {
    auto targetTriple = llvmModule.getTargetTriple();
    if (targetTriple.empty())
      targetTriple = llvm::sys::getDefaultTargetTriple();

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
      llvm::errs() << "Failed to lookup target: " << error << "\n";
      return failure();
    }

    llvm::TargetOptions opt;
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(targetTriple, "generic", "", opt, std::nullopt));
    llvmModule.setTargetTriple(targetTriple);
    llvmModule.setDataLayout(targetMachine->createDataLayout());

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

  LogicalResult linkExecutable(StringRef objectPath, StringRef runtimeLib,
    StringRef outputPath) {
    auto clangExe = llvm::sys::findProgramByName("clang++");
    if (!clangExe)
      clangExe = llvm::sys::findProgramByName("clang");
    if (!clangExe) {
      llvm::errs() << "error: clang++/clang executable not found in PATH\n";
      return failure();
    }

    std::string clangProgram = *clangExe;
    std::vector<std::string> argStorage;
    argStorage.push_back(clangProgram);
    argStorage.emplace_back(objectPath.str());
    argStorage.emplace_back(runtimeLib.str());
    argStorage.emplace_back("-O2");
    argStorage.emplace_back("-o");
    argStorage.emplace_back(outputPath.str());

    llvm::SmallVector<llvm::StringRef, 8> args;
    for (const auto& arg : argStorage)
      args.push_back(arg);

    std::string errorMessage;
    bool executionFailed = false;
    int result = llvm::sys::ExecuteAndWait(clangProgram, args, std::nullopt,
      std::nullopt, 0, 0, &errorMessage,
      &executionFailed);
    if (result != 0 || executionFailed) {
      if (!errorMessage.empty())
        llvm::errs() << errorMessage << "\n";
      llvm::errs() << "error: linking failed\n";
      return failure();
    }
    return success();
  }

  LogicalResult buildExecutable(llvm::Module& llvmModule, StringRef runtimeLib,
    StringRef outputPath) {
    llvm::SmallString<256> objectPath;
    if (auto ec = llvm::sys::fs::createTemporaryFile("lython", ".o", objectPath)) {
      llvm::errs() << "error: failed to create temporary object file: "
        << ec.message() << "\n";
      return failure();
    }
    llvm::FileRemover objCleanup(objectPath);
    if (failed(emitObjectFile(llvmModule, objectPath)))
      return failure();
    if (failed(linkExecutable(objectPath, runtimeLib, outputPath)))
      return failure();
    return success();
  }

  LogicalResult runJIT(ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybeEngine = mlir::ExecutionEngine::create(module);
    if (!maybeEngine) {
      llvm::errs() << "Failed to create ExecutionEngine\n";
      return failure();
    }
    auto engine = std::move(maybeEngine.get());
    registerRuntimeSymbols(*engine);
    int32_t exitCode = 0;
    if (auto err = engine->invoke("main", ExecutionEngine::result(exitCode))) {
      llvm::errs() << "JIT session error: " << err << "\n";
      llvm::errs() << "ExecutionEngine invocation failed\n";
      return failure();
    }
    return success();
  }

} // namespace

static llvm::cl::OptionCategory LythonCategory("lython options");
static llvm::cl::opt<std::string>
InputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
  llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string>
OutputFilename("o", llvm::cl::desc("Output file (default: a.out)"),
  llvm::cl::value_desc("filename"), llvm::cl::init("a.out"),
  llvm::cl::cat(LythonCategory));
static llvm::cl::opt<bool>
EmitLLVMOnly("emit-llvm",
  llvm::cl::desc("Stop after emitting LLVM IR to the output file"),
  llvm::cl::init(false), llvm::cl::cat(LythonCategory));
static llvm::cl::SubCommand JitCommand("jit", "JIT execute an input file");
static llvm::cl::opt<std::string>
JitInputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
  llvm::cl::Required, llvm::cl::sub(JitCommand));

int main(int argc, char** argv) {
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream& os) {
    os << "Lython CLI based on MLIR\n";
    });
  llvm::cl::ParseCommandLineOptions(
    argc, argv, "Lython compiler driver (clang-style)\n");

  const bool jitMode = static_cast<bool>(JitCommand);
  std::string inputPath;
  std::string outputPath = OutputFilename;

  if (jitMode) {
    inputPath = JitInputFilename;
  }
  else {
    if (InputFilename.empty()) {
      llvm::errs() << "error: no input files\n";
      llvm::cl::PrintHelpMessage();
      return 1;
    }
    inputPath = InputFilename;
  }

  auto buildRoot = findBuildRoot(argv[0]);
  if (!buildRoot) {
    llvm::errs() << "error: unable to locate CMake build root\n";
    return 1;
  }

  llvm::SmallString<256> runtimeLibPath(*buildRoot);
  llvm::sys::path::append(runtimeLibPath, "src", "lython", "runtime",
    "libLythonRuntime.a");
  if (!llvm::sys::fs::exists(runtimeLibPath)) {
    llvm::errs() << "error: runtime library not found at '"
      << runtimeLibPath << "'\n";
    return 1;
  }

  bool isPythonInput = llvm::sys::path::extension(inputPath) == ".py";
  std::optional<std::string> sourceRoot;
  if (isPythonInput) {
    sourceRoot = findSourceRoot(*buildRoot);
    if (!sourceRoot) {
      llvm::errs() << "error: unable to determine source root for Python frontend\n";
      return 1;
    }
  }

  DialectRegistry registry;
  registry.insert<py::PyDialect, func::FuncDialect, arith::ArithDialect,
    scf::SCFDialect, mlir::cf::ControlFlowDialect,
    LLVM::LLVMDialect>();
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);

  MLIRContext context;
  context.appendDialectRegistry(registry);

  std::string mlirSource;
  if (isPythonInput) {
    if (failed(generateMlirFromPython(inputPath, *sourceRoot, mlirSource)))
      return 1;
  }
  else {
    auto file = mlir::openInputFile(inputPath);
    if (!file) {
      llvm::errs() << "error: could not open input file '" << inputPath << "'\n";
      return 1;
    }
    mlirSource = std::string(file->getBuffer());
  }

  OwningOpRef<ModuleOp> module = parseModuleFromBuffer(mlirSource, context);
  if (!module)
    return 1;

  if (failed(runPipeline(*module, context))) {
    llvm::errs() << "Failed to run lowering pipeline\n";
    return 1;
  }

  if (jitMode) {
    return failed(runJIT(*module)) ? 1 : 0;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }

  if (EmitLLVMOnly)
    return failed(dumpLLVMIR(*llvmModule, outputPath)) ? 1 : 0;

  if (llvm::InitializeNativeTarget()) {
    llvm::errs() << "error: could not initialize native target\n";
    return 1;
  }
  if (llvm::InitializeNativeTargetAsmPrinter()) {
    llvm::errs() << "error: could not initialize native asm printer\n";
    return 1;
  }

  return failed(buildExecutable(*llvmModule, runtimeLibPath, outputPath)) ? 1 : 0;
}
