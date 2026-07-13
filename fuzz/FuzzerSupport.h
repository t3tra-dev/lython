#pragma once

#include "mlir/IR/DialectRegistry.h"

// Process-wide one-time setup shared by the harnesses: native target,
// embedded runtime module registration, and the full Lython dialect registry.
// The registry is reusable across inputs; each input gets a fresh MLIRContext.
const mlir::DialectRegistry &lythonFuzzerRegistry();
