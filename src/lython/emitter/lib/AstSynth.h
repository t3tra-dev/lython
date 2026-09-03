#pragma once

// Builders for SYNTHESIZED AST nodes.
//
// Desugaring is how this emitter implements most of what CPython does in C:
// a comprehension, a reducer, a `with`, an enum, a dataclass method and a
// fused iterator are all rewritten into ordinary Python AST and handed to the
// same emission path the parser's own tree takes. That means node building is
// not incidental to three or four files -- it is the shared vocabulary of the
// whole desugaring layer.
//
// ⭐ One vocabulary, not one per file. Three parallel families had grown
// (`nameNode`/`synthName`/`synthName`, `stringConstant`/`synthStr`/
// `synthStrConstant`, three FunctionDef builders differing only in whether
// they took annotations, defaults or decorators), plus hand-rolled
// `parser::makeNode` at ~150 call sites. Same nodes, three spellings, and a
// fix to one spelling reached none of the others.

#include "Ast.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace lython::emitter::synth {

using parser::NodePtr;
using parser::SourceRange;

// ---- expressions ----------------------------------------------------------

NodePtr name(llvm::StringRef id, SourceRange range);
NodePtr intConstant(std::int64_t value, SourceRange range);
NodePtr strConstant(llvm::StringRef text, SourceRange range);
NodePtr constantBool(bool value, SourceRange range);
NodePtr noneConstant(SourceRange range);
NodePtr attribute(NodePtr value, llvm::StringRef attr, SourceRange range);
NodePtr selfAttribute(llvm::StringRef receiver, llvm::StringRef attr,
                      SourceRange range);
NodePtr subscript(NodePtr value, NodePtr slice, SourceRange range);
NodePtr tuple(std::vector<NodePtr> elts, SourceRange range);
NodePtr call(NodePtr func, std::vector<NodePtr> args, SourceRange range);
// A `name=value` argument, and a call that carries some. The dispatcher's arms
// forward a keyword AS a keyword, because CPython binds it by name in the body
// that RUNS -- forwarding it by position would bind the subclass's parameter of
// that position instead.
NodePtr keyword(llvm::StringRef arg, NodePtr value, SourceRange range);
NodePtr callWithKeywords(NodePtr func, std::vector<NodePtr> args,
                         std::vector<NodePtr> keywords, SourceRange range);
NodePtr methodCall(NodePtr receiver, llvm::StringRef method,
                   std::vector<NodePtr> args, SourceRange range);
NodePtr lenCall(NodePtr value, SourceRange range);
NodePtr reprCall(NodePtr value, SourceRange range);
NodePtr binOp(NodePtr left, llvm::StringRef opKind, NodePtr right,
              SourceRange range);
NodePtr compare(NodePtr left, llvm::StringRef opKind, NodePtr right,
                SourceRange range);
NodePtr notOp(NodePtr operand, SourceRange range);
// A single value is returned as itself: `or` over one operand is that operand,
// and a one-element BoolOp is not a node the emitter expects.
NodePtr boolOp(llvm::StringRef opKind, std::vector<NodePtr> values,
               SourceRange range);
NodePtr ifExp(NodePtr test, NodePtr body, NodePtr orelse, SourceRange range);
// `x in y`, the spelling the dict/membership desugars build most.
NodePtr compareIn(NodePtr left, NodePtr right, SourceRange range);
// `a or b or ...`, collapsing a single operand (see boolOp).
NodePtr orChain(std::vector<NodePtr> values, SourceRange range);

// ---- statements -----------------------------------------------------------

NodePtr assign(NodePtr target, NodePtr value, SourceRange range);
NodePtr annAssign(NodePtr target, NodePtr annotation, NodePtr value,
                  SourceRange range);
NodePtr returnStmt(NodePtr value, SourceRange range);
NodePtr raiseStmt(NodePtr exc, SourceRange range);
// `raise <Exception>("<message>")`, the shape every synthesized guard raises.
NodePtr raiseCall(llvm::StringRef exception, llvm::StringRef message,
                  SourceRange range);
// `raise ValueError("<message>")`, by far the most raised of them.
NodePtr raiseValueError(llvm::StringRef message, SourceRange range);
NodePtr ifStmt(NodePtr test, std::vector<NodePtr> body,
               std::vector<NodePtr> orelse, SourceRange range);
NodePtr forStmt(NodePtr target, NodePtr iter, std::vector<NodePtr> body,
                std::vector<NodePtr> orelse, SourceRange range);
NodePtr whileStmt(NodePtr test, std::vector<NodePtr> body,
                  std::vector<NodePtr> orelse, SourceRange range);
// An expression evaluated as a statement (`Expr` wrapping it).
NodePtr exprStmt(NodePtr value, SourceRange range);
NodePtr breakStmt(SourceRange range);
NodePtr continueStmt(SourceRange range);
// `yield <value>` as a STATEMENT (an Expr wrapping the Yield).
NodePtr yieldStmt(NodePtr value, SourceRange range);

// ---- definitions ----------------------------------------------------------

struct Param {
  std::string name;
  NodePtr annotation;
};

NodePtr arg(llvm::StringRef name, NodePtr annotation, SourceRange range);

// The one FunctionDef builder. `paramNodesOut`, when given, receives the
// `arg` nodes in order -- the generator synthesis needs them to override
// parameter types after the def exists.
NodePtr functionDef(llvm::StringRef name, llvm::ArrayRef<Param> params,
                    std::vector<NodePtr> defaults, std::vector<NodePtr> body,
                    NodePtr returns, llvm::ArrayRef<llvm::StringRef> decorators,
                    SourceRange range,
                    llvm::SmallVectorImpl<const parser::Node *> *paramNodesOut =
                        nullptr);

} // namespace lython::emitter::synth
