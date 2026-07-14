#include "AstAccess.h"
#include "Driver.h"
#include "Parser.h"
#include "PyDialectTypes.h"
#include "TypeSystem.h"

#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

#include <memory>

namespace {

using lython::emitter::TypeSystem;
using lython::emitter::CallInferenceResult;
using lython::emitter::FunctionSignature;
namespace ast = lython::emitter::ast;

const mlir::DialectRegistry &testRegistry() {
  static mlir::DialectRegistry *registry = [] {
    auto *result = new mlir::DialectRegistry();
    lython::driver::registerLythonDialects(*result);
    return result;
  }();
  return *registry;
}

// Pins the observable behavior of the TypeSystem facade ahead of the
// Algorithm J migration: these tests describe what the current engine
// resolves, so behavior-preserving refactors keep them green.
class TypeSystemTest : public ::testing::Test {
protected:
  TypeSystemTest() : context(testRegistry()) {
    context.loadAllAvailableDialects();
    types = std::make_unique<TypeSystem>(context);
    types->seedBuiltins();
  }

  const lython::parser::Node &parseModule(llvm::StringRef source) {
    lython::parser::ParseOptions options;
    options.typeComments = true;
    parsed = std::make_unique<lython::parser::ParseResult>(
        lython::parser::parse(source, "<test>.py", options));
    EXPECT_TRUE(parsed->ok());
    return *parsed->tree;
  }

  const lython::parser::Node *statement(const lython::parser::Node &module,
                                        std::size_t index) {
    const auto *body = ast::nodeList(module, "body");
    if (!body || index >= body->size())
      return nullptr;
    return (*body)[index].get();
  }

  mlir::MLIRContext context;
  std::unique_ptr<TypeSystem> types;
  std::unique_ptr<lython::parser::ParseResult> parsed;
};

TEST_F(TypeSystemTest, AnnotationTypes) {
  const lython::parser::Node &module =
      parseModule("a: int\n"
                  "b: list[str]\n"
                  "c: dict[str, int]\n");
  const lython::parser::Node *a = statement(module, 0);
  const lython::parser::Node *b = statement(module, 1);
  const lython::parser::Node *c = statement(module, 2);
  ASSERT_TRUE(a && b && c);

  EXPECT_EQ(types->annotationType(ast::node(*a, "annotation")),
            types->intType());
  EXPECT_EQ(types->annotationType(ast::node(*b, "annotation")),
            types->listOf(types->strType()));
  EXPECT_EQ(types->annotationType(ast::node(*c, "annotation")),
            types->dictOf(types->strType(), types->intType()));
}

TEST_F(TypeSystemTest, InferConstantsWidenToBuiltins) {
  const lython::parser::Node &module = parseModule("x = 1\n"
                                                   "y = \"hi\"\n"
                                                   "z = 1.5\n"
                                                   "w = True\n");
  auto widened = [&](std::size_t index) {
    const lython::parser::Node *assign = statement(module, index);
    EXPECT_TRUE(assign);
    return types->widenLiteral(types->inferExpr(ast::node(*assign, "value")));
  };

  EXPECT_EQ(widened(0), types->intType());
  EXPECT_EQ(widened(1), types->strType());
  EXPECT_EQ(widened(2), types->floatType());
  EXPECT_EQ(widened(3), types->boolType());
}

TEST_F(TypeSystemTest, InferBinOpOnLiterals) {
  const lython::parser::Node &module = parseModule("x = 1 + 2\n");
  const lython::parser::Node *assign = statement(module, 0);
  ASSERT_TRUE(assign);
  mlir::Type result =
      types->widenLiteral(types->inferExpr(ast::node(*assign, "value")));
  EXPECT_EQ(result, types->intType());
}

TEST_F(TypeSystemTest, FunctionSignatureAnnotated) {
  const lython::parser::Node &module =
      parseModule("def f(a: int, b: str) -> str:\n"
                  "    return b\n");
  const lython::parser::Node *function = statement(module, 0);
  ASSERT_TRUE(function);

  FunctionSignature signature = types->functionSignature(*function);
  ASSERT_EQ(signature.positionalTypes.size(), 2u);
  EXPECT_EQ(signature.positionalTypes[0], types->intType());
  EXPECT_EQ(signature.positionalTypes[1], types->strType());
  EXPECT_EQ(signature.resultType, types->strType());
  EXPECT_TRUE(signature.missingParameterAnnotations.empty());
  EXPECT_TRUE(signature.invalidParameterAnnotations.empty());
  EXPECT_TRUE(signature.bodyInferenceFailures.empty());
  ASSERT_TRUE(signature.callable);
  ASSERT_TRUE(signature.publicCallable);
}

TEST_F(TypeSystemTest, FunctionSignatureInfersUnannotatedReturn) {
  const lython::parser::Node &module = parseModule("def f(a: int):\n"
                                                   "    return a\n");
  const lython::parser::Node *function = statement(module, 0);
  ASSERT_TRUE(function);

  FunctionSignature signature = types->functionSignature(*function);
  EXPECT_EQ(types->widenLiteral(signature.resultType), types->intType());
  EXPECT_TRUE(signature.missingParameterAnnotations.empty());
}

TEST_F(TypeSystemTest, FunctionSignatureRecordsMissingParameterAnnotation) {
  const lython::parser::Node &module = parseModule("def f(x):\n"
                                                   "    return x\n");
  const lython::parser::Node *function = statement(module, 0);
  ASSERT_TRUE(function);

  FunctionSignature signature = types->functionSignature(*function);
  ASSERT_EQ(signature.missingParameterAnnotations.size(), 1u);
  EXPECT_EQ(signature.missingParameterAnnotations[0], "x");
}

TEST_F(TypeSystemTest, GenericCallBindsTypeVariable) {
  const lython::parser::Node &module = parseModule("def ident[T](x: T) -> T:\n"
                                                   "    return x\n");
  const lython::parser::Node *function = statement(module, 0);
  ASSERT_TRUE(function);

  FunctionSignature signature = types->functionSignature(*function);
  ASSERT_TRUE(signature.publicCallable);

  CallInferenceResult result = types->inferCallWithEvidence(
      signature.publicCallable, {types->intType()}, {});
  ASSERT_TRUE(result);
  EXPECT_EQ(types->widenLiteral(result.resultType), types->intType());
  EXPECT_TRUE(result.evidence.callableContract);
}

TEST_F(TypeSystemTest, JoinTypes) {
  EXPECT_EQ(types->join({types->intType(), types->intType()}),
            types->intType());
  mlir::Type joined = types->join({types->intType(), types->strType()});
  ASSERT_TRUE(joined);
  EXPECT_NE(joined, types->intType());
  EXPECT_NE(joined, types->strType());
}

TEST_F(TypeSystemTest, WidenLiteral) {
  EXPECT_EQ(types->widenLiteral(types->literal("1")), types->intType());
  EXPECT_EQ(types->widenLiteral(types->literal("True")), types->boolType());
  EXPECT_EQ(types->widenLiteral(types->intType()), types->intType());
}

TEST_F(TypeSystemTest, LambdaAdoptsExpectedCallableParameterTypes) {
  const lython::parser::Node &module = parseModule("g = lambda x: x\n");
  const lython::parser::Node *assign = statement(module, 0);
  ASSERT_TRUE(assign);
  const lython::parser::Node *lambda = ast::node(*assign, "value");
  ASSERT_TRUE(lambda);
  ASSERT_EQ(lambda->kind, "Lambda");

  llvm::SmallVector<mlir::Type, 1> params{types->intType()};
  llvm::SmallVector<mlir::Type, 1> results{types->intType()};
  auto expected =
      py::CallableType::get(&context, params, {}, {}, {}, results);

  FunctionSignature signature =
      types->functionSignature(*lambda, std::nullopt, expected);
  ASSERT_EQ(signature.positionalTypes.size(), 1u);
  EXPECT_EQ(signature.positionalTypes[0], types->intType());
  EXPECT_EQ(types->widenLiteral(signature.resultType), types->intType());
}

} // namespace
