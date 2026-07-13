#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "PyDialect.h.inc"
#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES
