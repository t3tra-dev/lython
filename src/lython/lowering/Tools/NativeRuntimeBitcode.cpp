// Builds the native runtime for the machine this runs on and writes it as LLVM
// bitcode, with the triple and data layout it was built for beside it:
//
//   LythonNativeRuntimeBitcode -o runtime.bc --triple-out triple.txt
//                              --layout-out layout.txt
//
// lyc embeds the result and links it instead of lowering the runtime on every
// compile (RuntimeLibrary.cpp, linkEmbeddedNativeRuntime). The target is the
// one lyc's JIT detects, so the two agree on what "this machine" is; a compile
// for anything else does not use it.

#include "Common/RuntimeLibrary.h"
#include "embedded.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace {

llvm::cl::opt<std::string> Output("o", llvm::cl::Required,
                                  llvm::cl::desc("bitcode output"));
llvm::cl::opt<std::string> TripleOutput("triple-out", llvm::cl::Required,
                                        llvm::cl::desc("triple output"));
llvm::cl::opt<std::string> LayoutOutput("layout-out", llvm::cl::Required,
                                        llvm::cl::desc("data layout output"));

bool writeText(llvm::StringRef path, llvm::StringRef text) {
  std::error_code error;
  llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "error: cannot write " << path << ": " << error.message()
                 << "\n";
    return false;
  }
  out << text;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Lython native runtime bitcode builder\n");
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Exactly what lyc's JIT asks for its target machine, so the data layout
  // below is the one a JIT-compiled program's module carries.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) {
    llvm::errs() << "error: " << llvm::toString(builder.takeError()) << "\n";
    return 1;
  }
  auto machine = builder->createTargetMachine();
  if (!machine) {
    llvm::errs() << "error: " << llvm::toString(machine.takeError()) << "\n";
    return 1;
  }

  py::runtime_library::embedded::registerPyRuntimeEmbeddedModules();
  llvm::LLVMContext context;
  llvm::Module runtime("lython-native-runtime", context);
  runtime.setTargetTriple((*machine)->getTargetTriple());
  runtime.setDataLayout((*machine)->createDataLayout());
  if (mlir::failed(py::runtime_library::buildNativeRuntimeModule(runtime)))
    return 1;

  std::error_code error;
  llvm::raw_fd_ostream out(Output, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "error: cannot write " << Output << ": " << error.message()
                 << "\n";
    return 1;
  }
  llvm::WriteBitcodeToFile(runtime, out);
  if (!writeText(TripleOutput, runtime.getTargetTriple().str()) ||
      !writeText(LayoutOutput, runtime.getDataLayoutStr()))
    return 1;
  return 0;
}
