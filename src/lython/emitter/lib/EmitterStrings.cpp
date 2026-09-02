// Building a `builtins.str` at emit time: what `str()`/`format()`/`%`/an
// f-string field turn a value into, and the two template readers behind
// `"...".format(...)` and `"%s" % x`.
//
// Split out of EmitterCalls.cpp, where it sat because `str(x)` arrives as a
// call. What the file is about is the string, not the call: the format
// template readers, the conversion/spec pair, and the per-type stringify are
// one subject, and they used none of that file's call-shaping helpers.
#include "Contracts.h"
#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "ArithBuilders.h"
#include "ExceptionTaxonomy.h"
#include "PyProtocols.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"

#include "EmitterOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <optional>
#include <string>

namespace lython::emitter {

Value ModuleEmitter::emitEmptyStrConstant(const parser::Node &anchor) {
  mlir::Type literalType = types.literal("\"\"");
  auto op = py::StrConstantOp::create(builder, loc(anchor), literalType,
                                      builder.getStringAttr(""));
  return coerceValue(Value{op.getResult(), literalType},
                     types.contract("builtins.str"), anchor);
}

std::optional<Value>
ModuleEmitter::emitInheritedObjectStr(const parser::Node &anchor,
                                     Value receiver) {
  mlir::Type widened = types.widenLiteral(receiver.type);
  if (!inheritsObjectDefaultDunder(widened, "__str__"))
    return std::nullopt;
  if (std::optional<Value> dispatched =
          tryEmitClassDunder(anchor, receiver, "__repr__"))
    return *dispatched;
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(widened, "__repr__", {});
  if (!inference)
    return std::nullopt;
  mlir::Type strType = types.contract("builtins.str");
  Value value = coerceValue(receiver, widened, anchor);
  auto op = py::ReprOp::create(
      builder, loc(anchor), strType,
      mlir::FlatSymbolRefAttr::get(&context, "__repr__"),
      mlir::TypeAttr::get(callProtocolFor(inference)), value.value);
  return Value{op.getResult(), strType};
}

// Can `emitStringifyValue` below render this type? Asked WITHOUT emitting,
// which is why it is a separate body rather than a trial run: the union arm
// has to know every member is renderable before it commits to the branch
// chain, and a half-emitted chain is not something the caller can undo.
//
// It mirrors the ladder in `emitStringifyValue` condition for condition. The
// two drifting apart costs a refusal reported at the fall-through rather than
// at the member, not a wrong answer.
bool ModuleEmitter::canStringifyType(mlir::Type type) {
  mlir::Type widened = types.widenLiteral(type);
  if (widened == types.contract("builtins.str") || widened == types.none())
    return true;
  if (lookupClassMethod(widened, "__str__") ||
      lookupClassMethod(widened, "__repr__"))
    return true;
  if (isExceptionContractType(widened) &&
      types.inferMethodCallWithEvidence(widened, "__str__", {}))
    return true;
  return static_cast<bool>(
      types.inferMethodCallWithEvidence(widened, "__repr__", {}));
}

// ⭐ A UNION renders by TESTING ITS TAG, member by member, and rendering the
// one that is live. There is no other way to do it: the inactive members hold
// zeroed placeholders, so a `select` over eagerly-rendered arms would call
// `__repr__` on a header nobody wrote.
//
// The chain is right-nested -- test member 0, else test member 1, ... with the
// last member as the final else -- and each arm goes back through
// `emitStringifyValue`, so a member that is itself a class, an exception or a
// container renders the way it does anywhere else.
bool ModuleEmitter::canConvertType(mlir::Type type, int64_t conversion) {
  if (conversion == 's')
    return canStringifyType(type);
  mlir::Type widened = types.widenLiteral(type);
  if (widened == types.none())
    return true;
  if (lookupClassMethod(widened, "__repr__"))
    return true;
  if (!types.inferMethodCallWithEvidence(widened, "__repr__", {}))
    return false;
  if (conversion != 'a')
    return true;
  return static_cast<bool>(types.inferMethodCallWithEvidence(
      types.contract("builtins.str"), "__ascii__", {}));
}

Value ModuleEmitter::emitUnionMemberDispatch(
    const parser::Node &anchor, Value unionValue, py::UnionType unionType,
    mlir::Type resultType, llvm::function_ref<Value(Value)> perMember) {
  llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
  std::function<mlir::Value(unsigned)> arm = [&](unsigned index) -> mlir::Value {
    mlir::Type member = members[index];
    auto emitArm = [&]() -> mlir::Value {
      auto unwrap = py::UnionUnwrapOp::create(builder, loc(anchor), member,
                                              unionValue.value);
      return coerceValue(perMember(Value{unwrap.getResult(), member}),
                         resultType, anchor)
          .value;
    };
    if (index + 1 >= members.size())
      return emitArm();
    auto test = py::UnionTestOp::create(builder, loc(anchor),
                                        builder.getI1Type(), unionValue.value,
                                        mlir::TypeAttr::get(member));
    return emitValueDiamond(loc(anchor), test.getResult(), resultType, emitArm,
                            [&] { return arm(index + 1); });
  };
  return Value{arm(0), resultType};
}

Value ModuleEmitter::emitUnionStringify(const parser::Node &anchor, Value value,
                                        py::UnionType unionType,
                                        unsigned index, int64_t conversion) {
  mlir::Type strType = types.contract("builtins.str");
  llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
  auto renderMember = [&](mlir::Type member) -> mlir::Value {
    // None has no physical header to unwrap OR to dispatch on; its text is a
    // constant, the same answer the scalar ladder gives it.
    if (py::isPyNoneType(member))
      return emitStrLiteralPiece(anchor, "None").value;
    auto unwrap =
        py::UnionUnwrapOp::create(builder, loc(anchor), member, value.value);
    Value memberValue{unwrap.getResult(), member};
    std::optional<Value> text =
        conversion == 's'
            ? emitStringifyValue(anchor, memberValue)
            : emitConversionValue(anchor, memberValue, conversion);
    // Unreachable while `canStringifyType` gates every member, and cheaper to
    // answer than to prove: an empty string is a wrong ANSWER, so it may not
    // stand in. The caller checked; if the two ever disagree this is a crash
    // in a debug build rather than a silent blank.
    assert(text && "union member passed canConvertType but did not render");
    // ⛔ A str member renders to ITSELF (the first arm of
    // `emitStringifyValue` is the identity), so what comes back is the union's
    // own payload rather than a freshly built string the way every other
    // member's is. Retaining it here to even that out was tried and measured:
    // 42 B before and 42 B after, so the imbalance is not on this side. The
    // leak is in the callee (tests/probe/wb_union_str_member_render_leak.py).
    return coerceValue(*text, strType, anchor).value;
  };
  if (index + 1 >= members.size())
    return Value{renderMember(members[index]), strType};
  auto test =
      py::UnionTestOp::create(builder, loc(anchor), builder.getI1Type(),
                              value.value, mlir::TypeAttr::get(members[index]));
  mlir::Value rendered = emitValueDiamond(
      loc(anchor), test.getResult(), strType,
      [&] { return renderMember(members[index]); },
      [&] {
        return emitUnionStringify(anchor, value, unionType, index + 1,
                                  conversion)
            .value;
      });
  return Value{rendered, strType};
}

std::optional<Value> ModuleEmitter::emitStringifyValue(const parser::Node &anchor,
                                                       Value value) {
  mlir::Type strType = types.contract("builtins.str");
  mlir::Type valueType = types.widenLiteral(value.type);
  if (valueType == strType)
    return coerceValue(value, strType, anchor);
  // A union renders through its tag. Placed before the dunder ladder because
  // none of those questions has an answer for a union: it has no class, no
  // manifest contract and no header of its own to dispatch on, which is what
  // "unnarrowed ... cannot be used where a concrete object is required" was
  // reporting from the lowering when `print(d.get("b"))` fell through here.
  if (auto unionType = mlir::dyn_cast<py::UnionType>(valueType)) {
    llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
    if (members.empty())
      return std::nullopt;
    for (mlir::Type member : members)
      if (!canStringifyType(member))
        return std::nullopt;
    return emitUnionStringify(anchor, value, unionType, /*index=*/0);
  }
  // None has no physical header for a __repr__ dispatch to receive; its
  // text is a compile-time constant anyway.
  if (valueType == types.none())
    return emitStrLiteralPiece(anchor, "None");
  if (std::optional<Value> dispatched = tryEmitClassDunder(anchor, value, "__str__"))
    return *dispatched;
  // ⭐ An EXCEPTION is the one builtin whose str and repr differ: str is the
  // message, repr is `ClassName('message')`. Falling to __repr__ below gave
  // f-strings, format() and %s the repr:
  //
  //     e = ValueError("plain")
  //     print(f"{e}")      # printed ValueError('plain'); CPython prints plain
  //
  // `tryEmitStrCall` already answers this for the `str(e)` spelling, through
  // the same `isExceptionContractType` question; this is the other spelling
  // reaching the same answer rather than a second rule.
  //
  // ⛔ Why NOT after the source-class __repr__ below, where this stood: an
  // exception subclass that declares only __repr__ still INHERITS
  // BaseException.__str__, so `class E(ValueError)` with a `__repr__` and no
  // `__str__` printed E-repr for `f"{e}"` where CPython prints the message.
  // Reaching the taxonomy first is also what keeps the __repr__ override gate
  // away from exceptions, which never render through __repr__ here: a
  // base-typed exception whose subclass declares __repr__ was refused for a
  // dispatch this ladder does not make. The __str__ gate above still stands,
  // and it is the one that has to, because a subclass __str__ DOES win.
  //
  // An erased `builtins.object` joins for the same reason from the other
  // side: the manifest object __str__ dispatches the payload class and falls
  // back to the repr form exactly where CPython's str(x) does, so the erased
  // str inside `e.args` renders unquoted. `tryEmitStrCall` already asks this
  // (its isErasedObject arm); reaching __repr__ below instead is how
  // `print(e.args[0], "x")` printed `'zz' x` where single-argument print --
  // which reaches object.__str__ down in the lowering -- printed `zz`.
  auto isErasedObject = [&](mlir::Type type) {
    auto contractType = mlir::dyn_cast<py::ContractType>(type);
    return contractType && contractType.getContractName() == "builtins.object";
  };
  if (isExceptionContractType(valueType) || isErasedObject(valueType)) {
    if (CallInferenceResult strInference =
            types.inferMethodCallWithEvidence(valueType, "__str__", {})) {
      Value receiver = coerceValue(value, valueType, anchor);
      auto op = py::StrOp::create(
          builder, loc(anchor), strType,
          mlir::FlatSymbolRefAttr::get(&context, "__str__"),
          mlir::TypeAttr::get(callProtocolFor(strInference)), receiver.value);
      return Value{op.getResult(), strType};
    }
  }
  // A source class without __str__ stringifies through its own __repr__
  // (CPython object.__str__ delegates to type(x).__repr__); the manifest
  // evidence below would instead resolve the generic object formatter,
  // which cannot see source methods.
  if (std::optional<Value> dispatched = tryEmitClassDunder(anchor, value, "__repr__"))
    return *dispatched;
  // Non-str builtins render via __repr__ (str(x) == repr(x) for every
  // non-str builtin; __str__ evidence resolves for containers through the
  // object contract but the manifest only implements their __repr__).
  if (CallInferenceResult inference =
          types.inferMethodCallWithEvidence(valueType, "__repr__", {})) {
    Value receiver = coerceValue(value, valueType, anchor);
    auto op = py::ReprOp::create(
        builder, loc(anchor), strType,
        mlir::FlatSymbolRefAttr::get(&context, "__repr__"),
        mlir::TypeAttr::get(callProtocolFor(inference)), receiver.value);
    return Value{op.getResult(), strType};
  }
  return std::nullopt;
}

std::optional<Value>
ModuleEmitter::emitConversionValue(const parser::Node &anchor, Value value,
                                   int64_t conversion) {
  mlir::Type strType = types.contract("builtins.str");
  if (conversion == 's')
    return emitStringifyValue(anchor, value);
  // 'r' and 'a' both start from __repr__.
  mlir::Type valueType = types.widenLiteral(value.type);
  if (valueType == types.none())
    return emitStrLiteralPiece(anchor, "None");
  // ⭐ AND A UNION STILL RENDERS BY TAG HERE. The str direction has had this
  // arm since `print(d.get("b"))` needed it; the repr direction never grew
  // one, and every spelling that reaches it failed on a union -- `repr(u)`
  // and `str(u)` disagreeing about the same value:
  //
  //     repr(d.get("b"))   # unresolved name 'repr'
  //     f"{u!r}"           # f-string conversion is not statically resolvable
  //     f"{u!a}"           # the same
  //     "%r" % (u,)        # %-formatting conversion is not statically ...
  //     ascii(u)           # ascii() is not supported for this argument type
  //
  // None of those is a question about the union: each member has a __repr__,
  // and which one to ask is what the tag says. The member arms recurse through
  // THIS function rather than the str one, so a str member comes back quoted.
  //
  // ⛔ Gated on `canConvertType` and not `canStringifyType`: a source class
  // that declares only __str__ renders in the str direction and has nothing to
  // answer here, and the branch chain cannot be half-emitted and undone.
  if (auto unionType = mlir::dyn_cast<py::UnionType>(valueType)) {
    llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
    if (members.empty())
      return std::nullopt;
    for (mlir::Type member : members)
      if (!canConvertType(member, conversion))
        return std::nullopt;
    return emitUnionStringify(anchor, value, unionType, /*index=*/0,
                              conversion);
  }
  std::optional<Value> repr;
  if (dispatchIsUnresolvable(value, "__repr__", /*receiverNode=*/nullptr,
                             /*throughSuper=*/false)) {
    if (std::optional<Value> dispatched =
            tryEmitVirtualDispatchWithValues(anchor, value, "__repr__", {}))
      return *dispatched;
    if (refuseUnresolvableDispatch(anchor, value, "__repr__"))
      return emitNone(anchor);
  }
  if (std::optional<Value> dispatched =
          tryEmitClassDunder(anchor, value, "__repr__")) {
    repr = *dispatched;
  } else if (CallInferenceResult inference =
                 types.inferMethodCallWithEvidence(valueType, "__repr__", {})) {
    Value receiver = coerceValue(value, valueType, anchor);
    auto op = py::ReprOp::create(
        builder, loc(anchor), strType,
        mlir::FlatSymbolRefAttr::get(&context, "__repr__"),
        mlir::TypeAttr::get(callProtocolFor(inference)), receiver.value);
    repr = Value{op.getResult(), strType};
  }
  if (!repr)
    return std::nullopt;
  if (conversion != 'a')
    return repr;
  // ascii(): repr with non-ASCII code points escaped, a manifest primitive
  // on str.
  CallInferenceResult ascii =
      types.inferMethodCallWithEvidence(strType, "__ascii__", {});
  if (!ascii)
    return std::nullopt;
  Value receiver = coerceValue(*repr, strType, anchor);
  Value posPack = emitPack({});
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  auto op = py::CallOp::create(builder, loc(anchor), mlir::TypeRange{strType},
                               callProtocolFor(ascii), receiver.value,
                               posPack.value, namePack.value, valuePack.value);
  op->setAttr("ly.bound_method", builder.getStringAttr("__ascii__"));
  return Value{op.getResults().front(), strType};
}

Value ModuleEmitter::emitFormatValue(const parser::Node &anchor, Value value,
                                     std::optional<Value> spec,
                                     bool specKnownEmpty) {
  mlir::Type strType = types.contract("builtins.str");
  mlir::Type valueType = types.widenLiteral(value.type);
  if (refuseUnresolvableDispatch(anchor, value, "__format__"))
    return emitNone(anchor);
  // The presence check precedes the gate so that the empty format spec is
  // materialized only on the path that consumes it.
  if (lookupClassMethod(valueType, "__format__")) {
    Value specValue =
        spec ? coerceValue(*spec, strType, anchor) : emitEmptyStrConstant(anchor);
    if (std::optional<Value> formatted =
            tryEmitClassDunder(anchor, value, "__format__", {specValue}))
      return *formatted;
  }
  if (CallInferenceResult inference =
          types.inferMethodCallWithEvidence(valueType, "__format__",
                                            {strType})) {
    Value receiver = coerceValue(value, valueType, anchor);
    Value specValue =
        spec ? coerceValue(*spec, strType, anchor) : emitEmptyStrConstant(anchor);
    Value posPack = emitPack({specValue});
    Value namePack = emitPack({});
    Value valuePack = emitPack({});
    mlir::Type resultType = inference.resultType ? inference.resultType
                                                 : strType;
    auto op = py::CallOp::create(
        builder, loc(anchor), mlir::TypeRange{resultType},
        callProtocolFor(inference), receiver.value, posPack.value,
        namePack.value, valuePack.value);
    op->setAttr("ly.bound_method", builder.getStringAttr("__format__"));
    return {op.getResults().front(), resultType};
  }
  // CPython's object.__format__: str(self) for an empty spec, TypeError
  // otherwise. The TypeError side is a static boundary here.
  if (!spec || specKnownEmpty) {
    if (std::optional<Value> text = emitStringifyValue(anchor, value))
      return *text;
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      "static type does not provide '__format__' (a non-empty format "
      "specification needs a manifest or source-class __format__)"});
  return emitNone(anchor);
}

Value ModuleEmitter::emitFormattedValue(const parser::Node &expr) {
  const parser::Node *valueNode = ast::node(expr, "value");
  if (!valueNode || valueNode->kind == "Error") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "malformed f-string interpolation"});
    return emitNone(expr);
  }
  int64_t conversion = ast::integer(expr, "conversion").value_or(-1);
  const parser::Node *specNode = ast::node(expr, "format_spec");

  Value inner = emitExpr(valueNode);
  std::optional<Value> spec;
  bool specKnownEmpty = false;
  if (specNode) {
    // A literal-only spec is inspectable now: empty text keeps the
    // str()-fallback available (CPython: format(x, '') == str(x)).
    bool allLiteral = true;
    std::string literalText;
    if (const auto *parts = ast::nodeList(*specNode, "values")) {
      for (const parser::NodePtr &part : *parts) {
        if (part && part->kind == "Constant") {
          if (auto text = ast::string(*part, "value"))
            literalText += std::string(*text);
          continue;
        }
        allLiteral = false;
      }
    }
    specKnownEmpty = allLiteral && literalText.empty();
    if (!specKnownEmpty)
      spec = emitJoinedStr(*specNode);
  }

  Value subject = inner;
  if (conversion == 'r' || conversion == 's' || conversion == 'a') {
    std::optional<Value> converted =
        emitConversionValue(expr, inner, conversion);
    if (!converted) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "f-string conversion is not statically resolvable for this "
          "operand type"});
      return emitNone(expr);
    }
    subject = *converted;
  }
  return emitFormatValue(expr, subject, spec, specKnownEmpty);
}

Value ModuleEmitter::emitJoinedStr(const parser::Node &expr) {
  mlir::Type strType = types.contract("builtins.str");
  const auto *parts = ast::nodeList(expr, "values");
  std::optional<Value> accumulated;
  auto append = [&](Value piece) {
    Value coerced = coerceValue(piece, strType, expr);
    if (!accumulated) {
      accumulated = coerced;
      return;
    }
    accumulated = emitBinarySpecial<py::AddOp>(expr, "__add__", *accumulated,
                                               coerced, strType);
  };
  if (parts) {
    for (const parser::NodePtr &part : *parts) {
      if (!part)
        continue;
      if (part->kind == "Constant") {
        append(emitConstant(*part));
        continue;
      }
      if (part->kind == "FormattedValue") {
        append(emitFormattedValue(*part));
        continue;
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, part->range.start,
          "unsupported f-string component '" + part->kind + "'"});
    }
  }
  if (!accumulated)
    return emitEmptyStrConstant(expr);
  return *accumulated;
}

Value ModuleEmitter::emitStrLiteralPiece(const parser::Node &anchor,
                                         llvm::StringRef text) {
  mlir::Type literalType = types.literal("\"" + text.str() + "\"");
  auto op = py::StrConstantOp::create(builder, loc(anchor), literalType,
                                      builder.getStringAttr(text));
  return coerceValue(Value{op.getResult(), literalType},
                     types.contract("builtins.str"), anchor);
}

std::optional<Value>
ModuleEmitter::emitValueAttribute(const parser::Node &anchor, Value object,
                                  llvm::StringRef attr) {
  std::optional<mlir::Type> field = lookupClassField(object.type, attr);
  if (!field)
    return std::nullopt;
  auto op = py::AttrGetOp::create(builder, loc(anchor), *field, object.value,
                                  attr);
  op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  if (auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(object.type))
    op->setAttr("ly.attr.owner",
                builder.getStringAttr(contract.getContractName()));
  return Value{op.getResult(), *field};
}

std::optional<Value> ModuleEmitter::emitValueIndex(const parser::Node &anchor,
                                                   Value object,
                                                   llvm::StringRef indexText) {
  bool numeric = !indexText.empty() &&
                 llvm::all_of(indexText, [](char c) { return c >= '0' && c <= '9'; });
  Value index;
  if (numeric) {
    mlir::Type literalType = types.literal(indexText.str());
    auto op = py::IntConstantOp::create(builder, loc(anchor), literalType,
                                        builder.getStringAttr(indexText));
    index = Value{op.getResult(), literalType};
  } else {
    index = emitStrLiteralPiece(anchor, indexText);
  }
  mlir::Type containerType = types.widenLiteral(object.type);
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      containerType, "__getitem__", {types.widenLiteral(index.type)});
  if (!inference)
    return std::nullopt;
  Value receiver = coerceValue(object, containerType, anchor);
  auto op = py::GetItemOp::create(
      builder, loc(anchor), inference.resultType,
      mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
      callProtocolFor(inference), receiver.value, index.value);
  return Value{op.getResult(), inference.resultType};
}

namespace {
// One replacement field of a compile-time template: {name.path[i]!c:spec}.
struct TemplateField {
  std::string name;
  // Accessor chain after the name: (true, "attr") or (false, "index").
  llvm::SmallVector<std::pair<bool, std::string>, 2> path;
  int conversion = -1;
  std::string spec;
  bool hasSpec = false;
};

// Splits `text` into literal chunks (first) and fields (second non-null).
// Returns false on a malformed template.
bool parseFormatTemplate(
    llvm::StringRef text,
    llvm::SmallVectorImpl<std::pair<std::string, std::optional<TemplateField>>>
        &out,
    std::string &error) {
  std::string literal;
  size_t i = 0, n = text.size();
  auto flushLiteral = [&]() {
    if (!literal.empty()) {
      out.push_back({literal, std::nullopt});
      literal.clear();
    }
  };
  while (i < n) {
    char c = text[i];
    if (c == '{') {
      if (i + 1 < n && text[i + 1] == '{') {
        literal += '{';
        i += 2;
        continue;
      }
      // replacement field
      size_t j = i + 1;
      int depth = 1;
      size_t colon = std::string::npos, bang = std::string::npos;
      while (j < n && depth > 0) {
        char d = text[j];
        if (d == '{')
          ++depth;
        else if (d == '}')
          --depth;
        else if (d == ':' && depth == 1 && colon == std::string::npos)
          colon = j;
        else if (d == '!' && depth == 1 && colon == std::string::npos &&
                 bang == std::string::npos && j + 1 < n && text[j + 1] != '=' &&
                 text[j + 1] != '}')
          bang = j;
        if (depth > 0)
          ++j;
      }
      if (depth != 0) {
        error = "Single '{' encountered in format string";
        return false;
      }
      size_t end = j; // position of matching '}'
      TemplateField field;
      size_t nameEnd = bang != std::string::npos    ? bang
                       : colon != std::string::npos ? colon
                                                    : end;
      std::string nameText = text.substr(i + 1, nameEnd - i - 1).str();
      // split accessors
      size_t p = 0;
      while (p < nameText.size() && nameText[p] != '.' && nameText[p] != '[')
        ++p;
      field.name = nameText.substr(0, p);
      while (p < nameText.size()) {
        if (nameText[p] == '.') {
          size_t q = p + 1;
          while (q < nameText.size() && nameText[q] != '.' &&
                 nameText[q] != '[')
            ++q;
          field.path.push_back({true, nameText.substr(p + 1, q - p - 1)});
          p = q;
        } else if (nameText[p] == '[') {
          size_t q = nameText.find(']', p + 1);
          if (q == std::string::npos) {
            error = "Missing ']' in format string";
            return false;
          }
          field.path.push_back({false, nameText.substr(p + 1, q - p - 1)});
          p = q + 1;
        } else {
          error = "invalid replacement field name";
          return false;
        }
      }
      if (bang != std::string::npos) {
        size_t convEnd = colon != std::string::npos ? colon : end;
        if (convEnd != bang + 2) {
          error = "invalid conversion specifier";
          return false;
        }
        char conv = text[bang + 1];
        if (conv != 'r' && conv != 's' && conv != 'a') {
          error = std::string("invalid conversion character '") + conv +
                  "'; expected 's', 'r', or 'a'";
          return false;
        }
        field.conversion = conv;
      }
      if (colon != std::string::npos) {
        field.hasSpec = true;
        field.spec = text.substr(colon + 1, end - colon - 1).str();
      }
      flushLiteral();
      out.push_back({std::string(), field});
      i = end + 1;
      continue;
    }
    if (c == '}') {
      if (i + 1 < n && text[i + 1] == '}') {
        literal += '}';
        i += 2;
        continue;
      }
      error = "Single '}' encountered in format string";
      return false;
    }
    literal += c;
    ++i;
  }
  flushLiteral();
  return true;
}
} // namespace

std::optional<Value>
ModuleEmitter::tryEmitStrFormatCall(const parser::Node &expr,
                                    const parser::Node *calleeNode) {
  if (!calleeNode || calleeNode->kind != "Attribute")
    return std::nullopt;
  auto methodName = ast::string(*calleeNode, "attr");
  if (!methodName || *methodName != "format")
    return std::nullopt;
  const parser::Node *receiverNode = ast::node(*calleeNode, "value");
  if (!receiverNode)
    return std::nullopt;
  mlir::Type strType = types.contract("builtins.str");
  if (types.widenLiteral(types.inferExpr(receiverNode)) != strType)
    return std::nullopt;

  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  auto diag = [&](std::string message) -> std::optional<Value> {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::move(message)});
    return emitNone(expr);
  };
  if (args)
    for (const parser::NodePtr &argument : *args)
      if (argument && argument->kind == "Starred")
        return diag("str.format does not support * unpacking (pass the "
                    "arguments explicitly)");

  std::optional<std::string> tpl;
  if (receiverNode->kind == "Constant")
    if (auto text = ast::string(*receiverNode, "value"))
      tpl = std::string(*text);
  if (!tpl) {
    // Runtime template (R4): the field/argument matching that a literal
    // template gets at compile time moves into the __fmt_* scanners, which
    // only accept auto-numbered fields — anything needing runtime argument
    // selection stays a loud error.
    if (keywords && !keywords->empty())
      return diag("str.format with a runtime template supports positional "
                  "arguments only (named fields need a literal template)");
    Value receiver =
        coerceValue(emitExpr(receiverNode), strType, expr);
    llvm::SmallVector<Value, 4> runtimeArgs;
    if (args)
      for (const parser::NodePtr &argument : *args)
        runtimeArgs.push_back(emitExpr(argument.get()));
    auto callStrMethod = [&](llvm::StringRef method, Value self,
                             llvm::ArrayRef<Value> extra)
        -> std::optional<Value> {
      llvm::SmallVector<mlir::Type, 4> argTypes;
      for (const Value &argument : extra)
        argTypes.push_back(types.widenLiteral(argument.type));
      CallInferenceResult inference = types.inferMethodCallWithEvidence(
          types.widenLiteral(self.type), method, argTypes);
      if (!inference)
        return std::nullopt;
      Value posPack = emitPack(extra);
      Value namePack = emitPack({});
      Value valuePack = emitPack({});
      auto op = py::CallOp::create(
          builder, loc(expr), mlir::TypeRange{inference.resultType},
          callProtocolFor(inference), self.value, posPack.value,
          namePack.value, valuePack.value);
      op->setAttr("ly.bound_method", builder.getStringAttr(method));
      return Value{op.getResults().front(), inference.resultType};
    };
    mlir::Type zeroLiteral = types.literal("0");
    auto zeroOp = py::IntConstantOp::create(builder, loc(expr), zeroLiteral,
                                            builder.getStringAttr("0"));
    Value cursor = coerceValue(Value{zeroOp.getResult(), zeroLiteral},
                               types.intType(), expr);
    std::optional<Value> acc;
    auto append = [&](Value piece) {
      Value coerced = coerceValue(piece, strType, expr);
      if (!acc) {
        acc = coerced;
        return;
      }
      acc = emitBinarySpecial<py::AddOp>(expr, "__add__", *acc, coerced,
                                         strType);
    };
    for (Value &argument : runtimeArgs) {
      std::optional<Value> prefix =
          callStrMethod("__fmt_prefix__", receiver, {cursor});
      std::optional<Value> pos =
          callStrMethod("__fmt_next__", receiver, {cursor});
      if (!prefix || !pos)
        return diag("runtime str.format scanners are missing from the "
                    "runtime manifest");
      std::optional<Value> conv =
          callStrMethod("__fmt_conv__", receiver, {*pos});
      std::optional<Value> spec =
          callStrMethod("__fmt_spec__", receiver, {*pos});
      std::optional<Value> next =
          callStrMethod("__fmt_end__", receiver, {*pos});
      if (!conv || !spec || !next)
        return diag("runtime str.format scanners are missing from the "
                    "runtime manifest");
      cursor = *next;
      // The conversion character is runtime data, so every variant is
      // formatted eagerly and __fmt_pick__ selects; formatting is pure, so
      // the extra work is the price of static dispatch.
      Value plain = emitFormatValue(expr, argument, spec, false);
      std::optional<Value> reprText =
          emitConversionValue(expr, argument, 'r');
      std::optional<Value> strText = emitStringifyValue(expr, argument);
      std::optional<Value> asciiText =
          emitConversionValue(expr, argument, 'a');
      if (!reprText || !strText || !asciiText)
        return diag("runtime str.format arguments must support repr/str/"
                    "ascii conversions statically");
      Value reprFmt = emitFormatValue(expr, *reprText, spec, false);
      Value strFmt = emitFormatValue(expr, *strText, spec, false);
      Value asciiFmt = emitFormatValue(expr, *asciiText, spec, false);
      std::optional<Value> picked = callStrMethod(
          "__fmt_pick__", coerceValue(plain, strType, expr),
          {*conv, reprFmt, strFmt, asciiFmt});
      if (!picked)
        return diag("runtime str.format scanners are missing from the "
                    "runtime manifest");
      append(*prefix);
      append(*picked);
    }
    std::optional<Value> tail =
        callStrMethod("__fmt_tail__", receiver, {cursor});
    if (!tail)
      return diag("runtime str.format scanners are missing from the runtime "
                  "manifest");
    append(*tail);
    return acc ? *acc : emitEmptyStrConstant(expr);
  }

  llvm::SmallVector<std::pair<std::string, std::optional<TemplateField>>, 8>
      parts;
  std::string parseError;
  if (!parseFormatTemplate(*tpl, parts, parseError))
    return diag(parseError);

  // Evaluate every argument once, in source order (CPython evaluates the
  // arguments before formatting; repeated field references reuse values).
  llvm::SmallVector<Value, 4> positional;
  llvm::StringMap<Value> named;
  if (args)
    for (const parser::NodePtr &argument : *args)
      positional.push_back(emitExpr(argument.get()));
  if (keywords)
    for (const parser::NodePtr &keyword : *keywords) {
      if (!keyword)
        continue;
      auto name = ast::string(*keyword, "arg");
      const parser::Node *valueNode = ast::node(*keyword, "value");
      if (!name || !valueNode)
        return diag("str.format does not support ** unpacking");
      named[*name] = emitExpr(valueNode);
    }

  int autoIndex = 0;
  bool sawManual = false, sawAuto = false;
  auto resolveField = [&](const TemplateField &field)
      -> std::optional<Value> {
    Value base;
    if (field.name.empty()) {
      if (sawManual) {
        diag("cannot switch from manual field specification to automatic "
             "field numbering");
        return std::nullopt;
      }
      sawAuto = true;
      if (autoIndex >= static_cast<int>(positional.size())) {
        diag("Replacement index " + std::to_string(autoIndex) +
             " out of range for positional args tuple");
        return std::nullopt;
      }
      base = positional[autoIndex++];
    } else if (llvm::all_of(field.name,
                            [](char c) { return c >= '0' && c <= '9'; })) {
      if (sawAuto) {
        diag("cannot switch from automatic field numbering to manual field "
             "specification");
        return std::nullopt;
      }
      sawManual = true;
      unsigned index = 0;
      (void)llvm::StringRef(field.name).getAsInteger(10, index);
      if (index >= positional.size()) {
        diag("Replacement index " + field.name +
             " out of range for positional args tuple");
        return std::nullopt;
      }
      base = positional[index];
    } else {
      auto hit = named.find(field.name);
      if (hit == named.end()) {
        diag("str.format has no keyword argument '" + field.name + "'");
        return std::nullopt;
      }
      base = hit->second;
    }
    for (const auto &accessor : field.path) {
      std::optional<Value> next =
          accessor.first ? emitValueAttribute(expr, base, accessor.second)
                         : emitValueIndex(expr, base, accessor.second);
      if (!next) {
        diag("cannot statically resolve accessor '" +
             std::string(accessor.first ? "." : "[") + accessor.second +
             std::string(accessor.first ? "" : "]") +
             "' in str.format replacement field");
        return std::nullopt;
      }
      base = *next;
    }
    return base;
  };

  // A spec may itself contain replacement fields ({0:>{1}}); they resolve
  // against the same argument list and auto-numbering sequence.
  auto emitSpecValue = [&](const TemplateField &field)
      -> std::optional<std::pair<std::optional<Value>, bool>> {
    if (!field.hasSpec)
      return std::make_pair(std::optional<Value>(), false);
    if (field.spec.find('{') == std::string::npos) {
      if (field.spec.empty())
        return std::make_pair(std::optional<Value>(), true);
      return std::make_pair(
          std::optional<Value>(emitStrLiteralPiece(expr, field.spec)), false);
    }
    llvm::SmallVector<std::pair<std::string, std::optional<TemplateField>>, 4>
        specParts;
    std::string specError;
    if (!parseFormatTemplate(field.spec, specParts, specError)) {
      diag(specError);
      return std::nullopt;
    }
    std::optional<Value> acc;
    auto append = [&](Value piece) {
      Value coerced = coerceValue(piece, strType, expr);
      if (!acc) {
        acc = coerced;
        return;
      }
      acc = emitBinarySpecial<py::AddOp>(expr, "__add__", *acc, coerced,
                                         strType);
    };
    for (const auto &part : specParts) {
      if (!part.second) {
        append(emitStrLiteralPiece(expr, part.first));
        continue;
      }
      if (part.second->hasSpec || part.second->conversion != -1) {
        diag("nested replacement fields in a format spec cannot themselves "
             "carry conversions or specs");
        return std::nullopt;
      }
      std::optional<Value> nested = resolveField(*part.second);
      if (!nested)
        return std::nullopt;
      std::optional<Value> text = emitStringifyValue(expr, *nested);
      if (!text) {
        diag("nested format-spec field is not statically stringifiable");
        return std::nullopt;
      }
      append(*text);
    }
    if (!acc)
      return std::make_pair(std::optional<Value>(), true);
    return std::make_pair(acc, false);
  };

  std::optional<Value> result;
  auto appendResult = [&](Value piece) {
    Value coerced = coerceValue(piece, strType, expr);
    if (!result) {
      result = coerced;
      return;
    }
    result = emitBinarySpecial<py::AddOp>(expr, "__add__", *result, coerced,
                                          strType);
  };
  for (const auto &part : parts) {
    if (!part.second) {
      appendResult(emitStrLiteralPiece(expr, part.first));
      continue;
    }
    std::optional<Value> subject = resolveField(*part.second);
    if (!subject)
      return emitNone(expr);
    if (part.second->conversion != -1) {
      std::optional<Value> converted =
          emitConversionValue(expr, *subject, part.second->conversion);
      if (!converted)
        return diag("str.format conversion is not statically resolvable for "
                    "this operand type");
      subject = converted;
    }
    auto spec = emitSpecValue(*part.second);
    if (!spec)
      return emitNone(expr);
    appendResult(emitFormatValue(expr, *subject, spec->first, spec->second));
  }
  if (!result)
    return emitEmptyStrConstant(expr);
  return *result;
}

Value ModuleEmitter::emitPercentFormat(const parser::Node &expr) {
  mlir::Type strType = types.contract("builtins.str");
  const parser::Node *leftNode = ast::node(expr, "left");
  const parser::Node *rightNode = ast::node(expr, "right");
  auto diag = [&](std::string message) -> Value {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start, std::move(message)});
    return emitNone(expr);
  };
  std::optional<std::string> tpl;
  if (leftNode && leftNode->kind == "Constant")
    if (auto text = ast::string(*leftNode, "value"))
      tpl = std::string(*text);
  if (!tpl)
    return diag("%-formatting requires a compile-time format string here "
                "(runtime templates are limited to statically-typed forms, "
                "which are not implemented for this call shape yet)");

  // Argument list: a tuple literal supplies its elements, anything else is
  // the single argument. Values are emitted once, in order.
  llvm::SmallVector<Value, 4> arguments;
  if (rightNode && rightNode->kind == "Tuple") {
    if (const auto *elements = ast::nodeList(*rightNode, "elts"))
      for (const parser::NodePtr &element : *elements) {
        if (element && element->kind == "Starred")
          return diag("%-formatting does not support * unpacking");
        arguments.push_back(emitExpr(element.get()));
      }
  } else if (rightNode) {
    arguments.push_back(emitExpr(rightNode));
  }

  std::optional<Value> result;
  auto appendResult = [&](Value piece) {
    Value coerced = coerceValue(piece, strType, expr);
    if (!result) {
      result = coerced;
      return;
    }
    result = emitBinarySpecial<py::AddOp>(expr, "__add__", *result, coerced,
                                          strType);
  };

  const std::string &text = *tpl;
  std::string literal;
  size_t i = 0, n = text.size(), argIndex = 0;
  auto flushLiteral = [&]() {
    if (!literal.empty()) {
      appendResult(emitStrLiteralPiece(expr, literal));
      literal.clear();
    }
  };
  auto nextArgument = [&](char type) -> std::optional<Value> {
    if (argIndex >= arguments.size()) {
      diag("not enough arguments for format string");
      return std::nullopt;
    }
    (void)type;
    return arguments[argIndex++];
  };
  while (i < n) {
    char c = text[i];
    if (c != '%') {
      literal += c;
      ++i;
      continue;
    }
    if (i + 1 < n && text[i + 1] == '%') {
      literal += '%';
      i += 2;
      continue;
    }
    size_t j = i + 1;
    if (j < n && text[j] == '(')
      return diag("%-formatting with a mapping key ('%(name)s') needs a dict "
                  "argument, which is not statically supported; use "
                  "str.format or an f-string");
    std::string flags;
    while (j < n && (text[j] == '#' || text[j] == '0' || text[j] == '-' ||
                     text[j] == ' ' || text[j] == '+'))
      flags += text[j++];
    if (j < n && text[j] == '*')
      return diag("%-formatting with '*' width is not supported; interpolate "
                  "the width into the format string instead");
    std::string width;
    while (j < n && text[j] >= '0' && text[j] <= '9')
      width += text[j++];
    std::string precision;
    bool hasPrecision = false;
    if (j < n && text[j] == '.') {
      hasPrecision = true;
      ++j;
      if (j < n && text[j] == '*')
        return diag("%-formatting with '*' precision is not supported; "
                    "interpolate the precision into the format string "
                    "instead");
      while (j < n && text[j] >= '0' && text[j] <= '9')
        precision += text[j++];
      if (precision.empty())
        precision = "0";
    }
    while (j < n && (text[j] == 'h' || text[j] == 'l' || text[j] == 'L'))
      ++j; // length modifiers are accepted and ignored, like CPython
    if (j >= n)
      return diag("incomplete format specifier at end of format string");
    char type = text[j];
    ++j;

    bool minus = flags.find('-') != std::string::npos;
    bool zero = flags.find('0') != std::string::npos && !minus;
    bool alt = flags.find('#') != std::string::npos;
    char sign = flags.find('+') != std::string::npos    ? '+'
                : flags.find(' ') != std::string::npos ? ' '
                                                       : '\0';
    auto numericSpec = [&](char specType, bool wantPrecision) {
      std::string spec;
      if (minus)
        spec += '<';
      if (sign)
        spec += sign;
      if (alt && (specType == 'o' || specType == 'x' || specType == 'X' ||
                  specType == 'e' || specType == 'E' || specType == 'f' ||
                  specType == 'F' || specType == 'g' || specType == 'G'))
        spec += '#';
      if (zero)
        spec += '0';
      spec += width;
      if (wantPrecision && hasPrecision)
        spec += "." + precision;
      spec += specType;
      return spec;
    };

    flushLiteral();
    switch (type) {
    case 'd':
    case 'i':
    case 'u': {
      if (hasPrecision)
        return diag("%-formatting precision on integer conversions is not "
                    "supported; use zero padding ('%0Nd') instead");
      std::optional<Value> argument = nextArgument(type);
      if (!argument)
        return emitNone(expr);
      // ⭐ A FLOAT operand is truncated toward zero first. CPython's `%d`
      // goes through `PyNumber_Long`, so `"%d" % (3.7,)` is 3 and
      // `"%d" % (-3.7,)` is -3; handing the float to `__format__` with a 'd'
      // code instead died at run time with "Unknown format code 'd' for
      // object of type 'float'".
      Value integral = *argument;
      if (types.widenLiteral(integral.type) == types.floatType()) {
        mlir::Type intContract = py::CallableType::get(
            &context, {types.floatType()}, {}, {}, {}, {types.intType()});
        auto truncated = py::IntOp::create(
            builder, loc(expr), types.intType(),
            mlir::FlatSymbolRefAttr::get(&context, "__int__"),
            mlir::TypeAttr::get(intContract), integral.value);
        integral = Value{truncated.getResult(), types.intType()};
      }
      // Always pass the 'd' spec: with an empty spec a bool would render
      // through str() as True/False, but %d must print 1/0.
      std::string spec = numericSpec('d', false);
      appendResult(emitFormatValue(
          expr, integral,
          std::optional<Value>(emitStrLiteralPiece(expr, spec)), false));
      break;
    }
    case 'o':
    case 'x':
    case 'X':
    case 'e':
    case 'E':
    case 'f':
    case 'F':
    case 'g':
    case 'G': {
      std::optional<Value> argument = nextArgument(type);
      if (!argument)
        return emitNone(expr);
      bool isFloatType = type == 'e' || type == 'E' || type == 'f' ||
                         type == 'F' || type == 'g' || type == 'G';
      std::string spec = numericSpec(type, isFloatType);
      appendResult(emitFormatValue(
          expr, *argument,
          std::optional<Value>(emitStrLiteralPiece(expr, spec)), false));
      break;
    }
    case 'c': {
      std::optional<Value> argument = nextArgument(type);
      if (!argument)
        return emitNone(expr);
      mlir::Type argType = types.widenLiteral(argument->type);
      if (argType == strType) {
        // CPython requires a length-1 str; a literal is checkable now.
        appendResult(coerceValue(*argument, strType, expr));
      } else {
        std::string spec = numericSpec('c', false);
        appendResult(emitFormatValue(
            expr, *argument,
            std::optional<Value>(emitStrLiteralPiece(expr, spec)), false));
      }
      break;
    }
    case 's':
    case 'r':
    case 'a': {
      std::optional<Value> argument = nextArgument(type);
      if (!argument)
        return emitNone(expr);
      std::optional<Value> textValue =
          type == 's' ? emitStringifyValue(expr, *argument)
                      : emitConversionValue(expr, *argument, type);
      if (!textValue)
        return diag("%-formatting conversion is not statically resolvable "
                    "for this operand type");
      if (width.empty() && !hasPrecision) {
        appendResult(*textValue);
        break;
      }
      std::string spec;
      spec += minus ? '<' : '>';
      spec += width;
      if (hasPrecision)
        spec += "." + precision;
      appendResult(emitFormatValue(
          expr, *textValue,
          std::optional<Value>(emitStrLiteralPiece(expr, spec)), false));
      break;
    }
    default:
      return diag(std::string("unsupported format character '") + type +
                  "' in %-format string");
    }
    i = j;
  }
  flushLiteral();
  if (argIndex != arguments.size())
    return diag("not all arguments converted during string formatting");
  if (!result)
    return emitEmptyStrConstant(expr);
  return *result;
}

std::optional<Value>
ModuleEmitter::tryEmitFormatCall(const parser::Node &expr,
                                 const parser::Node *calleeNode) {
  if (!callsUnshadowedBuiltin(calleeNode, "format"))
    return std::nullopt;
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->empty() || args->size() > 2 ||
      (keywords && !keywords->empty()))
    return std::nullopt;
  Value value = emitExpr(args->front().get());
  std::optional<Value> spec;
  bool specKnownEmpty = false;
  if (args->size() == 2) {
    const parser::Node *specNode = (*args)[1].get();
    mlir::Type specType = types.widenLiteral(types.inferExpr(specNode));
    if (specType != types.contract("builtins.str")) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, specNode->range.start,
          "format() format_spec must be statically typed str"});
      return emitNone(expr);
    }
    if (specNode->kind == "Constant") {
      if (auto text = ast::string(*specNode, "value"))
        specKnownEmpty = text->empty();
    }
    spec = emitExpr(specNode);
  }
  return emitFormatValue(expr, value, spec, specKnownEmpty);
}

} // namespace lython::emitter
