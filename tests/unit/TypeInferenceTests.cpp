#include "Driver.h"
#include "PyDialectTypes.h"
#include "TypeInference.h"

#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace {

using lython::emitter::InferenceContext;
using VarKind = InferenceContext::VarKind;

const mlir::DialectRegistry &testRegistry() {
  static mlir::DialectRegistry *registry = [] {
    auto *result = new mlir::DialectRegistry();
    lython::driver::registerLythonDialects(*result);
    return result;
  }();
  return *registry;
}

class TypeInferenceTest : public ::testing::Test {
protected:
  TypeInferenceTest() : context(testRegistry()) {
    context.loadAllAvailableDialects();
  }

  mlir::Type intType() {
    return py::ContractType::get(&context, "builtins.int");
  }
  mlir::Type strType() {
    return py::ContractType::get(&context, "builtins.str");
  }
  mlir::Type listOf(mlir::Type element) {
    return py::ContractType::get(&context, "builtins.list", {element});
  }
  mlir::Type dictOf(mlir::Type key, mlir::Type value) {
    return py::ContractType::get(&context, "builtins.dict", {key, value});
  }

  mlir::MLIRContext context;
};

TEST_F(TypeInferenceTest, FreshVariablesAreDistinct) {
  InferenceContext inference(context);
  mlir::Type a = inference.freshVar(VarKind::Inference);
  mlir::Type b = inference.freshVar(VarKind::Inference);
  EXPECT_NE(a, b);
  EXPECT_TRUE(py::isPyInferVarType(a));
  EXPECT_FALSE(py::isPyContractType(a));
  EXPECT_FALSE(py::isPyType(a));
}

TEST_F(TypeInferenceTest, UnifyBindsVariableToGround) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference);
  ASSERT_TRUE(inference.unify(alpha, intType()));
  EXPECT_EQ(inference.zonk(alpha), intType());
  EXPECT_TRUE(inference.fullyResolved(alpha));
}

TEST_F(TypeInferenceTest, UnifyIsTransitiveThroughVariables) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference);
  mlir::Type beta = inference.freshVar(VarKind::Inference);
  ASSERT_TRUE(inference.unify(alpha, beta));
  ASSERT_TRUE(inference.unify(beta, intType()));
  EXPECT_EQ(inference.zonk(alpha), intType());
  EXPECT_EQ(inference.zonk(beta), intType());
}

TEST_F(TypeInferenceTest, UnifyDecomposesContractArguments) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference);
  ASSERT_TRUE(inference.unify(listOf(alpha), listOf(intType())));
  EXPECT_EQ(inference.zonk(alpha), intType());
}

TEST_F(TypeInferenceTest, UnifyGroundMismatchFails) {
  InferenceContext inference(context);
  InferenceContext::UnifyResult result = inference.unify(intType(), strType());
  EXPECT_FALSE(result);
  EXPECT_NE(result.reason.find("cannot unify"), std::string::npos);
}

TEST_F(TypeInferenceTest, UnifyArityMismatchFails) {
  InferenceContext inference(context);
  EXPECT_FALSE(inference.unify(listOf(intType()),
                               dictOf(intType(), intType())));
}

TEST_F(TypeInferenceTest, OccursCheckRejectsInfiniteType) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference, nullptr, "'xs'");
  InferenceContext::UnifyResult result = inference.unify(alpha, listOf(alpha));
  EXPECT_FALSE(result);
  EXPECT_NE(result.reason.find("infinite type"), std::string::npos);
  EXPECT_NE(result.reason.find("'xs'"), std::string::npos);
}

TEST_F(TypeInferenceTest, OccursCheckSeesThroughBindings) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference);
  mlir::Type beta = inference.freshVar(VarKind::Inference);
  ASSERT_TRUE(inference.unify(beta, listOf(alpha)));
  EXPECT_FALSE(inference.unify(alpha, listOf(beta)));
}

TEST_F(TypeInferenceTest, ZonkKeepsUnboundVariables) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference);
  mlir::Type beta = inference.freshVar(VarKind::Inference);
  ASSERT_TRUE(inference.unify(alpha, intType()));
  mlir::Type zonked = inference.zonk(dictOf(alpha, beta));
  EXPECT_FALSE(inference.fullyResolved(zonked));
  EXPECT_TRUE(py::containsPyInferVar(zonked));
  auto contract = mlir::cast<py::ContractType>(zonked);
  EXPECT_EQ(contract.getArguments()[0], intType());
  EXPECT_TRUE(py::isPyInferVarType(contract.getArguments()[1]));
}

TEST_F(TypeInferenceTest, UnifyCallableStructurally) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Inference);
  mlir::Type beta = inference.freshVar(VarKind::Inference);
  llvm::SmallVector<mlir::Type, 1> paramsA{alpha};
  llvm::SmallVector<mlir::Type, 1> resultsA{beta};
  llvm::SmallVector<mlir::Type, 1> paramsB{intType()};
  llvm::SmallVector<mlir::Type, 1> resultsB{strType()};
  mlir::Type callableA =
      py::CallableType::get(&context, paramsA, {}, {}, {}, resultsA);
  mlir::Type callableB =
      py::CallableType::get(&context, paramsB, {}, {}, {}, resultsB);
  ASSERT_TRUE(inference.unify(callableA, callableB));
  EXPECT_EQ(inference.zonk(alpha), intType());
  EXPECT_EQ(inference.zonk(beta), strType());
}

TEST_F(TypeInferenceTest, GeneralizeQuantifiesAndInstantiateIsFresh) {
  InferenceContext inference(context);
  mlir::Type scheme;
  {
    InferenceContext::LevelScope scope(inference);
    mlir::Type alpha = inference.freshVar(VarKind::Inference);
    scheme = inference.generalize(0, listOf(alpha));
  }
  auto contract = mlir::cast<py::ContractType>(scheme);
  ASSERT_EQ(contract.getArguments().size(), 1u);
  auto quantified =
      mlir::dyn_cast<py::TypeVarType>(contract.getArguments()[0]);
  ASSERT_TRUE(quantified);
  EXPECT_TRUE(quantified.getName().starts_with(
      InferenceContext::generalizedPrefix));

  mlir::Type first = inference.instantiate(scheme);
  mlir::Type second = inference.instantiate(scheme);
  EXPECT_NE(first, second);
  auto firstArg =
      mlir::cast<py::ContractType>(first).getArguments()[0];
  ASSERT_TRUE(py::isPyInferVarType(firstArg));
  ASSERT_TRUE(inference.unify(firstArg, intType()));
  auto secondArg =
      mlir::cast<py::ContractType>(second).getArguments()[0];
  ASSERT_TRUE(inference.unify(secondArg, strType()));
  EXPECT_EQ(inference.zonk(firstArg), intType());
  EXPECT_EQ(inference.zonk(secondArg), strType());
}

TEST_F(TypeInferenceTest, LevelPropagationBlocksEscapedGeneralization) {
  InferenceContext inference(context);
  mlir::Type outer = inference.freshVar(VarKind::Inference);
  mlir::Type generalized;
  {
    InferenceContext::LevelScope scope(inference);
    mlir::Type inner = inference.freshVar(VarKind::Inference);
    // Binding the outer variable to a structure containing the inner one
    // lowers the inner variable's level: it escaped into the environment.
    ASSERT_TRUE(inference.unify(outer, listOf(inner)));
    generalized = inference.generalize(0, inner);
  }
  EXPECT_TRUE(py::isPyInferVarType(generalized))
      << "escaped variable must not be quantified";
}

TEST_F(TypeInferenceTest, MergedVariablesShareBinding) {
  InferenceContext inference(context);
  mlir::Type alpha = inference.freshVar(VarKind::Instantiation);
  mlir::Type beta = inference.freshVar(VarKind::Inference);
  ASSERT_TRUE(inference.unify(alpha, beta));
  auto alphaVar = mlir::cast<py::InferVarType>(alpha);
  // Inference strictness survives merges in either direction.
  EXPECT_EQ(inference.varKind(alphaVar.getId()), VarKind::Inference);
  ASSERT_TRUE(inference.unify(beta, strType()));
  EXPECT_EQ(inference.zonk(alpha), strType());
}

} // namespace
