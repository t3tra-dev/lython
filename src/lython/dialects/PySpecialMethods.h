#ifndef LYTHON_PY_SPECIAL_METHODS_H
#define LYTHON_PY_SPECIAL_METHODS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>

namespace py {

enum class SpecialMethodKind {
  GetItem,
  SetItem,
  DelItem,
  Iter,
  Next,
  Len,
  Repr,
  Str,
  Contains,
  Bool,
  Round,
  Enter,
  Exit,
  AEnter,
  AExit,
  AIter,
  ANext,
};

enum class SpecialMethodProtocolLowering {
  Dedicated,
  Generic,
};

enum class SpecialMethodOpLowering {
  CallableEvidence,
  Iterator,
  ConcreteMethod,
  MethodCall,
};

struct SpecialMethodDescriptor {
  llvm::StringLiteral name;
  SpecialMethodKind kind;
  std::size_t minArgumentCount;
  std::size_t maxArgumentCount;
  llvm::ArrayRef<llvm::StringRef> operandNames;
  SpecialMethodProtocolLowering protocolLowering;
  SpecialMethodOpLowering opLowering;
  bool isContext;
  bool isEnter;
  bool isAsync;

  bool acceptsArgumentCount(std::size_t count) const {
    return count >= minArgumentCount && count <= maxArgumentCount;
  }
};

inline llvm::ArrayRef<SpecialMethodDescriptor> specialMethodDescriptors() {
  static const llvm::StringRef receiverOperands[] = {"receiver"};
  static const llvm::StringRef indexedOperands[] = {"receiver", "index"};
  static const llvm::StringRef indexedValueOperands[] = {"receiver", "index",
                                                         "value"};
  static const llvm::StringRef containsOperands[] = {"receiver", "item"};
  static const llvm::StringRef roundOperands[] = {"receiver", "ndigits"};
  static const llvm::StringRef contextManagerOperands[] = {"manager"};
  static const llvm::StringRef contextExitOperands[] = {
      "manager", "exc_type", "exc_value", "traceback"};

  static const SpecialMethodDescriptor descriptors[] = {
      {"__getitem__", SpecialMethodKind::GetItem, 1, 1, indexedOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__setitem__", SpecialMethodKind::SetItem, 2, 2, indexedValueOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__delitem__", SpecialMethodKind::DelItem, 1, 1, indexedOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__iter__", SpecialMethodKind::Iter, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::Iterator, false, false, false},
      {"__next__", SpecialMethodKind::Next, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::Iterator, false, false, false},
      {"__len__", SpecialMethodKind::Len, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__repr__", SpecialMethodKind::Repr, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__str__", SpecialMethodKind::Str, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__contains__", SpecialMethodKind::Contains, 1, 1, containsOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__bool__", SpecialMethodKind::Bool, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::CallableEvidence, false, false, false},
      {"__round__", SpecialMethodKind::Round, 0, 1, roundOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::ConcreteMethod, false, false, false},
      {"__enter__", SpecialMethodKind::Enter, 0, 0, contextManagerOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::ConcreteMethod, true, true, false},
      {"__exit__", SpecialMethodKind::Exit, 3, 3, contextExitOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::ConcreteMethod, true, false, false},
      {"__aenter__", SpecialMethodKind::AEnter, 0, 0, contextManagerOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::ConcreteMethod, true, true, true},
      {"__aexit__", SpecialMethodKind::AExit, 3, 3, contextExitOperands,
       SpecialMethodProtocolLowering::Generic,
       SpecialMethodOpLowering::ConcreteMethod, true, false, true},
      {"__aiter__", SpecialMethodKind::AIter, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::MethodCall, false, false, false},
      {"__anext__", SpecialMethodKind::ANext, 0, 0, receiverOperands,
       SpecialMethodProtocolLowering::Dedicated,
       SpecialMethodOpLowering::MethodCall, false, false, true},
  };
  return descriptors;
}

inline const SpecialMethodDescriptor *
specialMethodDescriptor(llvm::StringRef methodName) {
  for (const SpecialMethodDescriptor &descriptor : specialMethodDescriptors())
    if (methodName == descriptor.name)
      return &descriptor;
  return nullptr;
}

inline const SpecialMethodDescriptor *
contextSpecialMethodDescriptor(llvm::StringRef methodName) {
  const SpecialMethodDescriptor *descriptor =
      specialMethodDescriptor(methodName);
  if (!descriptor || !descriptor->isContext)
    return nullptr;
  return descriptor;
}

inline bool
usesGenericProtocolSpecialMethodLowering(llvm::StringRef methodName) {
  const SpecialMethodDescriptor *descriptor =
      specialMethodDescriptor(methodName);
  return descriptor &&
         descriptor->protocolLowering == SpecialMethodProtocolLowering::Generic;
}

inline bool supportsProtocolSpecialMethodLowering(llvm::StringRef methodName) {
  const SpecialMethodDescriptor *descriptor =
      specialMethodDescriptor(methodName);
  return descriptor &&
         descriptor->opLowering != SpecialMethodOpLowering::MethodCall;
}

} // namespace py

#endif // LYTHON_PY_SPECIAL_METHODS_H
