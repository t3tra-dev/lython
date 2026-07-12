// Layer-1 dispatch: one `lower*` method per
// py op, selected by TypeSwitch. Add new op lowerings here, not as rewrite
// patterns -- they need the RuntimeBundle evidence state.
#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

mlir::LogicalResult RuntimeBundleLowerer::lowerPyOp(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
      .Case<py::ClassOp>([&](auto classOp) {
        erase.push_back(classOp.getOperation());
        return mlir::success();
      })
      .Case<py::StrConstantOp>([&](auto str) { return lowerStrConstant(str); })
      .Case<py::IntConstantOp>(
          [&](auto integer) { return lowerIntConstant(integer); })
      .Case<py::FloatConstantOp>(
          [&](auto floating) { return lowerFloatConstant(floating); })
      .Case<py::BoolConstantOp>(
          [&](auto boolean) { return lowerBoolConstant(boolean); })
      .Case<py::NoneOp>([&](auto none) { return lowerNone(none); })
      .Case<py::CastFromPrimOp>(
          [&](auto cast) { return lowerCastFromPrim(cast); })
      .Case<py::TypeObjectOp>(
          [&](auto typeObject) { return lowerTypeObject(typeObject); })
      .Case<py::ClassUpcastOp, py::ClassRefineOp, py::ProtocolViewOp>(
          [&](auto view) { return lowerAliasViewOp(view); })
      .Case<py::ClassTestOp>([&](auto test) { return lowerClassTest(test); })
      .Case<py::UnionWrapOp>([&](auto wrap) { return lowerUnionWrap(wrap); })
      .Case<py::UnionTestOp>([&](auto test) { return lowerUnionTest(test); })
      .Case<py::UnionUnwrapOp>(
          [&](auto unwrap) { return lowerUnionUnwrap(unwrap); })
      .Case<py::AttrGetOp>([&](auto attr) { return lowerAttrGet(attr); })
      .Case<py::AttrSetOp>([&](auto attr) { return lowerAttrSet(attr); })
      .Case<py::GlobalGetOp>([&](auto get) { return lowerGlobalGet(get); })
      .Case<py::GlobalSetOp>([&](auto set) { return lowerGlobalSet(set); })
      .Case<py::PackOp>([&](auto pack) { return lowerPack(pack); })
      .Case<py::BindingRefOp>(
          [&](auto binding) { return lowerBindingRef(binding); })
      .Case<py::NewOp>([&](auto newOp) { return lowerNew(newOp); })
      .Case<py::InitOp>([&](auto init) { return lowerInit(init); })
      .Case<py::RaiseOp>([&](auto raise) { return lowerRaise(raise); })
      .Case<py::RaiseCurrentOp>(
          [&](auto raiseCurrent) { return lowerRaiseCurrent(raiseCurrent); })
      .Case<py::ExceptMatchOp>(
          [&](auto match) { return lowerExceptMatch(match); })
      .Case<py::ExceptCurrentMatchOp>(
          [&](auto match) { return lowerExceptCurrentMatch(match); })
      .Case<py::ExceptCurrentValueOp>(
          [&](auto value) { return lowerExceptCurrentValue(value); })
      .Case<py::CallOp>([&](auto call) { return lowerCall(call); })
      .Case<py::BoolOp>([&](auto boolean) { return lowerBool(boolean); })
      .Case<py::LenOp>([&](auto len) { return lowerLen(len); })
      .Case<py::GetItemOp>([&](auto getItem) { return lowerGetItem(getItem); })
      .Case<py::IterOp>([&](auto iter) { return lowerIter(iter); })
      .Case<py::NextOp>([&](auto next) { return lowerNext(next); })
      .Case<py::EnterOp>([&](auto enter) { return lowerEnter(enter); })
      .Case<py::ExitOp>([&](auto exit) { return lowerExit(exit); })
      .Case<py::AEnterOp>([&](auto enter) { return lowerAEnter(enter); })
      .Case<py::AExitOp>([&](auto exit) { return lowerAExit(exit); })
      .Case<py::AIterOp>([&](auto iter) { return lowerAIter(iter); })
      .Case<py::ANextOp>([&](auto next) { return lowerANext(next); })
      .Case<py::AwaitOp>([&](auto await) { return lowerAwait(await); })
      .Case<py::YieldValueOp>([&](auto yield) {
        yield.emitError() << "generator function lowering is not implemented "
                             "yet";
        return mlir::failure();
      })
      .Case<py::YieldFromOp>([&](auto yieldFrom) {
        yieldFrom.emitError()
            << "generator yield-from lowering is not implemented yet";
        return mlir::failure();
      })
      .Case<py::SetItemOp>([&](auto setItem) { return lowerSetItem(setItem); })
      .Case<py::DelItemOp>([&](auto delItem) { return lowerDelItem(delItem); })
      .Case<py::ContainsOp>(
          [&](auto contains) { return lowerContains(contains); })
      .Case<py::RoundOp>([&](auto round) { return lowerRound(round); })
      .Case<py::IncRefOp>([&](auto incRef) { return lowerIncRef(incRef); })
      .Case<py::DecRefOp>([&](auto decRef) { return lowerDecRef(decRef); })
      .Case<py::NegOp, py::PosOp, py::InvertOp>(
          [&](auto unary) { return lowerUnaryMethodOp(unary); })
      .Case<py::AddOp, py::SubOp, py::MulOp, py::DivOp, py::FloorDivOp,
            py::ModOp, py::LShiftOp, py::RShiftOp, py::BitAndOp, py::BitOrOp,
            py::BitXorOp, py::EqOp, py::NeOp, py::LtOp, py::LeOp, py::GtOp,
            py::GeOp>([&](auto binary) { return lowerBinaryMethodOp(binary); })
      .Case<py::ReprOp>(
          [&](auto repr) { return lowerNamedUnaryMethodOp(repr, "__repr__"); })
      .Case<py::StrOp>(
          [&](auto str) { return lowerNamedUnaryMethodOp(str, "__str__"); })
      .Default([&](mlir::Operation *unknown) {
        unknown->emitError()
            << "resolved Py op has no runtime lowering rule yet";
        return mlir::failure();
      });
}

} // namespace py::lowering
