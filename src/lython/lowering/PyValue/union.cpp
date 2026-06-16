#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <functional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

using StaticClassLayout = class_layout::Layout;

// A union lowers to an i64 active-member tag followed by the concatenated
// payload parts of its non-None members in normalized member order. This keeps
// the representation independent of nullable object headers, so primitive
// members such as bool can participate in unions without a separate dialect
// primitive.

const union_abi::MemberSlice *
findSlice(llvm::ArrayRef<union_abi::MemberSlice> slices,
          mlir::Type memberType) {
  for (const union_abi::MemberSlice &slice : slices)
    if (slice.memberType == memberType)
      return &slice;
  return nullptr;
}

// Builds all-null descriptor parts for one member through the LLVM zero
// bridge; the reconcile pass cancels the casts against the descriptor
// unpacking on the consumer side.
mlir::LogicalResult appendNullParts(mlir::Location loc, mlir::Type memberType,
                                    const PyLLVMTypeConverter &typeConverter,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    llvm::SmallVectorImpl<mlir::Value> &parts) {
  llvm::SmallVector<mlir::Type> partTypes;
  if (mlir::failed(typeConverter.convertType(memberType, partTypes)) ||
      partTypes.empty())
    return mlir::failure();
  for (mlir::Type partType : partTypes) {
    mlir::Type lowered = typeConverter.convertType(partType);
    if (!lowered)
      return mlir::failure();
    mlir::Value zero = rewriter.create<mlir::LLVM::ZeroOp>(loc, lowered);
    if (lowered == partType) {
      parts.push_back(zero);
      continue;
    }
    parts.push_back(
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                loc, mlir::TypeRange{partType}, mlir::ValueRange{zero})
            .getResult(0));
  }
  return mlir::success();
}

mlir::Value tagConstant(mlir::Location loc, unsigned tag,
                        mlir::ConversionPatternRewriter &rewriter) {
  return rewriter.create<mlir::arith::ConstantIntOp>(loc, tag, 64);
}

mlir::Value memberActiveBit(mlir::Location loc, mlir::Value tag,
                            UnionType unionType, mlir::Type memberType,
                            mlir::ConversionPatternRewriter &rewriter) {
  std::optional<unsigned> member = union_abi::memberTag(unionType, memberType);
  if (!member || !tag || !tag.getType().isInteger(64))
    return {};
  return rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, tag,
      tagConstant(loc, *member, rewriter));
}

mlir::FailureOr<mlir::Value>
remapUnionTag(mlir::Location loc, UnionType inputUnion, UnionType outputUnion,
              mlir::Value inputTag, mlir::ConversionPatternRewriter &rewriter,
              bool outputIsSubsetProjection = false) {
  if (!inputTag || !inputTag.getType().isInteger(64))
    return mlir::failure();

  mlir::Value result;
  llvm::ArrayRef<mlir::Type> sourceMembers = outputIsSubsetProjection
                                                 ? outputUnion.getMemberTypes()
                                                 : inputUnion.getMemberTypes();
  for (mlir::Type memberType : sourceMembers) {
    std::optional<unsigned> inputMemberTag =
        union_abi::memberTag(inputUnion, memberType);
    std::optional<unsigned> outputMemberTag =
        union_abi::memberTag(outputUnion, memberType);
    if (!inputMemberTag || !outputMemberTag)
      return mlir::failure();
    mlir::Value mapped = tagConstant(loc, *outputMemberTag, rewriter);
    if (!result) {
      result = mapped;
      continue;
    }
    mlir::Value active = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, inputTag,
        tagConstant(loc, *inputMemberTag, rewriter));
    result =
        rewriter.create<mlir::arith::SelectOp>(loc, active, mapped, result);
  }
  if (!result)
    return mlir::failure();
  return result;
}

mlir::LogicalResult appendClassUpcastParts(
    mlir::Location loc, mlir::Operation *op, ClassType sourceType,
    ClassType targetType, const union_abi::MemberSlice &slice,
    mlir::ValueRange inputParts, const PyLLVMTypeConverter &typeConverter,
    mlir::ConversionPatternRewriter &rewriter,
    llvm::SmallVectorImpl<mlir::Value> &parts) {
  (void)loc;
  (void)op;
  (void)sourceType;
  (void)targetType;
  (void)rewriter;
  llvm::SmallVector<mlir::Type, 2> targetParts;
  if (mlir::failed(typeConverter.convertType(targetType, targetParts)) ||
      targetParts.empty() || slice.count != targetParts.size())
    return mlir::failure();
  for (unsigned i = 0; i < slice.count; ++i) {
    mlir::Value part = inputParts[slice.offset + i];
    if (part.getType() != targetParts[i])
      return mlir::failure();
    parts.push_back(part);
  }
  return mlir::success();
}

struct ClassUnionCandidate {
  mlir::Value live;
  llvm::SmallVector<mlir::Value, 8> parts;
};

struct UnionWrapLowering : public mlir::OpConversionPattern<UnionWrapOp> {
  UnionWrapLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<UnionWrapOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(UnionWrapOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto unionType = mlir::cast<UnionType>(op.getResult().getType());
    mlir::Type inputType = op.getInput().getType();

    llvm::SmallVector<union_abi::MemberSlice> slices;
    if (mlir::failed(
            union_abi::memberPartSlices(*typeConverter, unionType, slices)))
      return rewriter.notifyMatchFailure(op, "failed to slice union parts");

    llvm::SmallVector<union_abi::MemberSlice> inputSlices;
    auto inputUnion = mlir::dyn_cast<UnionType>(inputType);
    if (inputUnion && mlir::failed(union_abi::memberPartSlices(
                          *typeConverter, inputUnion, inputSlices)))
      return rewriter.notifyMatchFailure(op, "failed to slice input union");

    mlir::ValueRange inputParts = adaptor.getInput();
    llvm::SmallVector<mlir::Value> parts;
    if (inputUnion) {
      if (inputParts.empty())
        return rewriter.notifyMatchFailure(op, "input union is missing tag");
      mlir::FailureOr<mlir::Value> mappedTag = remapUnionTag(
          op.getLoc(), inputUnion, unionType, inputParts.front(), rewriter);
      if (mlir::failed(mappedTag))
        return rewriter.notifyMatchFailure(op, "failed to remap union tag");
      parts.push_back(*mappedTag);
    } else {
      std::optional<unsigned> tag = union_abi::memberTag(unionType, inputType);
      if (!tag)
        return rewriter.notifyMatchFailure(op, "input type is not a member");
      parts.push_back(tagConstant(op.getLoc(), *tag, rewriter));
    }
    for (const union_abi::MemberSlice &slice : slices) {
      if (slice.count == 0)
        continue; // None member carries no parts.
      if (slice.memberType == inputType) {
        if (inputParts.size() != slice.count)
          return rewriter.notifyMatchFailure(op, "member part count mismatch");
        parts.append(inputParts.begin(), inputParts.end());
        continue;
      }
      if (inputUnion) {
        if (const union_abi::MemberSlice *inputSlice =
                findSlice(inputSlices, slice.memberType)) {
          for (unsigned i = 0; i < inputSlice->count; ++i)
            parts.push_back(inputParts[inputSlice->offset + i]);
          continue;
        }
      }
      if (mlir::failed(appendNullParts(op.getLoc(), slice.memberType,
                                       *typeConverter, rewriter, parts)))
        return rewriter.notifyMatchFailure(op, "failed to build null parts");
    }
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(parts)});
    return mlir::success();
  }
};

struct UnionUnwrapLowering : public mlir::OpConversionPattern<UnionUnwrapOp> {
  UnionUnwrapLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<UnionUnwrapOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(UnionUnwrapOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto unionType = mlir::cast<UnionType>(op.getInput().getType());
    mlir::Type resultType = op.getResult().getType();

    if (mlir::isa<NoneType>(resultType)) {
      mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
      if (!module)
        return mlir::failure();
      RuntimeAPI runtime(module, rewriter, *typeConverter);
      rewriter.replaceOp(op, runtime.getNoneValue(op.getLoc()));
      return mlir::success();
    }

    llvm::SmallVector<union_abi::MemberSlice> slices;
    if (mlir::failed(
            union_abi::memberPartSlices(*typeConverter, unionType, slices)))
      return rewriter.notifyMatchFailure(op, "failed to slice union parts");
    mlir::ValueRange inputParts = adaptor.getInput();
    if (inputParts.empty())
      return rewriter.notifyMatchFailure(op, "input union is missing tag");
    mlir::Value inputTag = inputParts.front();

    auto appendSlice = [&](const union_abi::MemberSlice &slice,
                           llvm::SmallVectorImpl<mlir::Value> &parts) {
      for (unsigned i = 0; i < slice.count; ++i)
        parts.push_back(inputParts[slice.offset + i]);
    };

    llvm::SmallVector<mlir::Value> parts;
    if (auto resultUnion = mlir::dyn_cast<UnionType>(resultType)) {
      // Subset projection remaps the active tag and keeps the projected
      // members' payload slices.
      mlir::FailureOr<mlir::Value> projectedTag =
          remapUnionTag(op.getLoc(), unionType, resultUnion, inputTag, rewriter,
                        /*outputIsSubsetProjection=*/true);
      if (mlir::failed(projectedTag))
        return rewriter.notifyMatchFailure(op, "failed to project union tag");
      parts.push_back(*projectedTag);
      for (mlir::Type member : resultUnion.getMemberTypes()) {
        const union_abi::MemberSlice *slice = findSlice(slices, member);
        if (!slice)
          return rewriter.notifyMatchFailure(op, "missing member slice");
        appendSlice(*slice, parts);
      }
    } else if (auto resultClass = mlir::dyn_cast<ClassType>(resultType)) {
      bool allMembersConform = true;
      llvm::SmallVector<ClassUnionCandidate, 4> candidates;
      for (const union_abi::MemberSlice &slice : slices) {
        if (slice.count == 0)
          continue;
        auto memberClass = mlir::dyn_cast<ClassType>(slice.memberType);
        if (!memberClass ||
            !isSubtypeOf(memberClass, resultClass, op.getOperation())) {
          allMembersConform = false;
          break;
        }

        mlir::Value live = memberActiveBit(op.getLoc(), inputTag, unionType,
                                           slice.memberType, rewriter);
        if (!live)
          return rewriter.notifyMatchFailure(
              op, "class union unwrap requires an active-member tag");
        ClassUnionCandidate candidate;
        candidate.live = live;
        if (mlir::failed(appendClassUpcastParts(
                op.getLoc(), op.getOperation(), memberClass, resultClass, slice,
                inputParts, *typeConverter, rewriter, candidate.parts)))
          return rewriter.notifyMatchFailure(
              op, "failed to build class upcast parts");
        candidates.push_back(std::move(candidate));
      }

      if (allMembersConform && !candidates.empty()) {
        if (candidates.size() == 1) {
          parts.append(candidates.front().parts.begin(),
                       candidates.front().parts.end());
        } else {
          llvm::SmallVector<mlir::Type, 8> resultTypes;
          if (mlir::failed(
                  typeConverter->convertType(resultType, resultTypes)) ||
              resultTypes.empty())
            return rewriter.notifyMatchFailure(
                op, "failed to convert class unwrap result type");

          std::function<llvm::SmallVector<mlir::Value, 8>(std::size_t)>
              selectCandidate = [&](std::size_t index) {
                if (index + 1 == candidates.size())
                  return candidates[index].parts;

                auto ifOp = rewriter.create<mlir::scf::IfOp>(
                    op.getLoc(), resultTypes, candidates[index].live,
                    /*withElseRegion=*/true);
                {
                  mlir::OpBuilder::InsertionGuard guard(rewriter);
                  rewriter.setInsertionPointToStart(ifOp.thenBlock());
                  rewriter.create<mlir::scf::YieldOp>(op.getLoc(),
                                                      candidates[index].parts);
                }
                {
                  mlir::OpBuilder::InsertionGuard guard(rewriter);
                  rewriter.setInsertionPointToStart(ifOp.elseBlock());
                  llvm::SmallVector<mlir::Value, 8> elseParts =
                      selectCandidate(index + 1);
                  rewriter.create<mlir::scf::YieldOp>(op.getLoc(), elseParts);
                }
                llvm::SmallVector<mlir::Value, 8> selected;
                selected.append(ifOp.getResults().begin(),
                                ifOp.getResults().end());
                return selected;
              };

          parts = selectCandidate(0);
        }
      } else {
        const union_abi::MemberSlice *slice = findSlice(slices, resultType);
        if (slice) {
          appendSlice(*slice, parts);
        } else {
          for (const union_abi::MemberSlice &candidate : slices) {
            auto memberClass = mlir::dyn_cast<ClassType>(candidate.memberType);
            if (!memberClass ||
                !isSubtypeOf(resultClass, memberClass, op.getOperation()))
              continue;
            if (mlir::failed(appendClassUpcastParts(
                    op.getLoc(), op.getOperation(), memberClass, resultClass,
                    candidate, inputParts, *typeConverter, rewriter, parts)))
              return rewriter.notifyMatchFailure(
                  op, "failed to build class downcast parts");
            break;
          }
          if (parts.empty())
            return rewriter.notifyMatchFailure(op, "missing member slice");
        }
      }
    } else {
      const union_abi::MemberSlice *slice = findSlice(slices, resultType);
      if (!slice)
        return rewriter.notifyMatchFailure(op, "missing member slice");
      appendSlice(*slice, parts);
    }
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(parts)});
    return mlir::success();
  }
};

struct UnionTestLowering : public mlir::OpConversionPattern<UnionTestOp> {
  UnionTestLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<UnionTestOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(UnionTestOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto unionType = mlir::cast<UnionType>(op.getInput().getType());
    llvm::SmallVector<union_abi::MemberSlice> slices;
    if (mlir::failed(
            union_abi::memberPartSlices(*typeConverter, unionType, slices)))
      return rewriter.notifyMatchFailure(op, "failed to slice union parts");
    mlir::ValueRange parts = adaptor.getInput();
    if (parts.empty())
      return rewriter.notifyMatchFailure(op, "input union is missing tag");
    mlir::Value tag = parts.front();

    mlir::Value live =
        memberActiveBit(op.getLoc(), tag, unionType, op.getMember(), rewriter);
    if (!live)
      return rewriter.notifyMatchFailure(
          op, "union member test requires an active-member tag");
    rewriter.replaceOp(op, live);
    return mlir::success();
  }
};

// py.len is emitted only after the frontend resolves a typing.mlir __len__
// contract. Lowering implements that proven operation from the concrete ABI:
// Unicode codepoint length for str, or the size slot for tuple/list/dict
// headers. The result re-enters the object world through LyLong_FromI64.
struct LenLowering : public mlir::OpConversionPattern<LenOp> {
  LenLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<LenOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(LenOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::Type inputType = op.getInput().getType();

    mlir::ValueRange parts = adaptor.getInput();
    mlir::Value length;
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    if (mlir::isa<StrType>(inputType)) {
      if (parts.size() != 2)
        return rewriter.notifyMatchFailure(op, "unexpected str parts");
      length = runtime
                   .call(op.getLoc(), RuntimeSymbols::kUnicodeCodepointLength,
                         rewriter.getI64Type(), parts)
                   .getResult();
    } else {
      int64_t sizeSlot = 0;
      int64_t refcountSlot = 0;
      unsigned lockComponent = 0;
      bool requiresLock = false;
      if (mlir::isa<TupleType>(inputType)) {
        sizeSlot = kTypedTupleSizeSlot;
        refcountSlot = kTypedTupleRefcountSlot;
      } else if (mlir::isa<ListType>(inputType)) {
        sizeSlot = kTypedListSizeSlot;
        refcountSlot = kTypedListRefcountSlot;
        lockComponent = kListLockComponent;
        requiresLock = true;
      } else if (mlir::isa<DictType>(inputType)) {
        sizeSlot = kTypedDictSizeSlot;
        refcountSlot = kTypedDictRefcountSlot;
        lockComponent = kDictLockComponent;
        requiresLock = true;
      } else {
        return rewriter.notifyMatchFailure(
            op, "py.len lowering supports str and managed tuple/list/dict "
                "containers");
      }

      if (parts.empty() || (requiresLock && parts.size() <= lockComponent))
        return rewriter.notifyMatchFailure(op, "unexpected container parts");
      mlir::Value header = parts[0];
      mlir::Value sizeIndex =
          createIndexConstant(op.getLoc(), rewriter, sizeSlot);
      if (requiresLock) {
        mlir::Value lock = parts[lockComponent];
        mlir::Value isManaged = container::Managed::predicate(
            op.getLoc(), header, refcountSlot, rewriter);
        auto ifOp = rewriter.create<mlir::scf::IfOp>(
            op.getLoc(), mlir::TypeRange{rewriter.getI64Type()}, isManaged,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());
          container::Managed::lock(op.getLoc(), header, lock, rewriter);
          auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
              op.getLoc(), header, sizeIndex);
          container::access::Contract::mark(sizeLoad.getOperation(), header,
                                            header);
          container::Managed::unlock(op.getLoc(), header, lock, rewriter);
          rewriter.create<mlir::scf::YieldOp>(op.getLoc(),
                                              sizeLoad.getResult());
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.elseBlock());
          auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
              op.getLoc(), header, sizeIndex);
          rewriter.create<mlir::scf::YieldOp>(op.getLoc(),
                                              sizeLoad.getResult());
        }
        length = ifOp.getResult(0);
      } else {
        length = rewriter.create<mlir::memref::LoadOp>(op.getLoc(), header,
                                                       sizeIndex);
      }
    }

    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            typeConverter->convertType(op.getResult().getType(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert py.int");
    auto boxed =
        runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                     mlir::TypeRange(resultTypes), mlir::ValueRange{length});
    llvm::SmallVector<mlir::Value> boxedParts(boxed.getResults().begin(),
                                              boxed.getResults().end());
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(boxedParts)});
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::union_::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  patterns.add<UnionWrapLowering, UnionUnwrapLowering, UnionTestLowering,
               LenLowering>(typeConverter, patterns.getContext());
}
} // namespace lowering::value::union_::Patterns

} // namespace py
