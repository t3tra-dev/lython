//===- python_mlir_opt.cpp - Python Dialect用mlir-opt ------------------===//
//
// Python Dialect を組み込んだ mlir-opt
//
//===----------------------------------------------------------------------===//

#include "src/dialects/include/Lython/PythonDialect/PythonDialect.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    
    // 標準MLIRダイアレクトを登録
    mlir::registerAllDialects(registry);
    
    // Python Dialectを登録
    registry.insert<mlir::python::PythonDialect>();
    
    // 標準パスを登録
    mlir::registerAllPasses();
    
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Python Dialect MLIR optimizer driver\n", registry));
}
