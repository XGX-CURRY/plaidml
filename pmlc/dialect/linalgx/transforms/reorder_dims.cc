// Copyright 2021 Intel Corporation

#include "pmlc/dialect/linalgx/transforms/reorder_dims.h"
#include "pmlc/dialect/linalgx/transforms/pass_detail.h"

namespace pmlc::dialect::linalgx {

using namespace mlir; // NOLINT

// For a 4d affine map, check if the 1st dimension is simple and the 3rd
// dimension is a complex expression.
static bool checkDim1and3(AffineMap map) {
  if (map.getNumResults() != 4) {
    return false;
  }
  auto dim1 = map.getResult(1);
  if (!dim1.isa<AffineDimExpr>() && !dim1.isa<AffineConstantExpr>()) {
    return false;
  }
  return map.getResult(3).isa<AffineBinaryOpExpr>();
}

static void reorderDim1and3(linalg::GenericOp op, ArrayRef<unsigned> argIdxs) {
  if (argIdxs.empty()) {
    return;
  }
  OpBuilder builder(op->getParentOp());
  auto context = op.getContext();
  auto loc = op.getLoc();
  auto numInputs = op.getNumInputs();

  SmallVector<Value, 4> newInputs = llvm::to_vector<4>(op.inputs());
  SmallVector<Value, 4> newOutputs = llvm::to_vector<4>(op.outputs());
  SmallVector<mlir::AffineMap, 4> newIdxMaps =
      llvm::to_vector<4>(op.indexing_maps().getAsValueRange<AffineMapAttr>());

  for (auto argIdx : argIdxs) {
    auto arg = op.getOperand(argIdx);
    auto type = arg.getType().dyn_cast<RankedTensorType>();
    auto shape = type.getShape();
    auto numDims = shape.size();
    SmallVector<int64_t, 4> newShape = {shape[0], shape[2], shape[3], shape[1]};
    auto elementType = type.getElementType();
    auto newType = RankedTensorType::get(newShape, elementType);
    auto idxMap = newIdxMaps[argIdx];
    auto exprs = idxMap.getResults();
    SmallVector<AffineExpr, 4> newExprs = {exprs[0], exprs[2], exprs[3],
                                           exprs[1]};
    auto newIdxMap = AffineMap::get(idxMap.getNumDims(), idxMap.getNumSymbols(),
                                    newExprs, context);
    newIdxMaps[argIdx] = newIdxMap;

    // Insert the a generic op for copy/reordering
    auto origMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    auto reorderMap = AffineMap::get(
        numDims, 0,
        ArrayRef{getAffineDimExpr(0, context), getAffineDimExpr(2, context),
                 getAffineDimExpr(3, context), getAffineDimExpr(1, context)},
        context);

    if (argIdx < op.getNumInputs()) {
      // The target argument is an input
      builder.setInsertionPoint(op);
      auto newArg =
          builder.create<linalg::InitTensorOp>(loc, newShape, elementType);

      // Before the original generic op, reorder the layout of the original
      // argument
      auto before = builder.create<linalg::GenericOp>(
          loc,
          /*resultTensorTypes=*/TypeRange{newType},
          /*inputs=*/ValueRange{arg},
          /*outputs=*/ValueRange{newArg},
          /*indexingMaps=*/ArrayRef<AffineMap>{origMap, reorderMap},
          /*iteratorTypes=*/SmallVector<StringRef, 4>(numDims, "parallel"),
          /*doc=*/"",
          /*libraryCall=*/"",
          [&](OpBuilder &builder, Location loc, ValueRange args) {
            builder.create<linalg::YieldOp>(loc, ValueRange{args[0]});
          });
      newInputs[argIdx] = before.getResult(0);
    } else {
      // The target argument is an output
      unsigned outIdx = argIdx - numInputs;
      builder.setInsertionPoint(op);
      newOutputs[outIdx] =
          builder.create<linalg::InitTensorOp>(loc, newShape, elementType);

      // After the original generic op, reorder the layout of the original
      // argument
      builder.setInsertionPointAfter(op);
      auto after = builder.create<linalg::GenericOp>(
          loc,
          /*resultTensorTypes=*/TypeRange{type},
          /*inputs=*/ValueRange{op.getResult(outIdx)},
          /*outputs=*/ValueRange{arg},
          /*indexingMaps=*/ArrayRef<AffineMap>{reorderMap, origMap},
          /*iteratorTypes=*/SmallVector<StringRef, 4>(numDims, "parallel"),
          /*doc=*/"",
          /*libraryCall=*/"",
          [&](OpBuilder &builder, Location loc, ValueRange args) {
            builder.create<linalg::YieldOp>(loc, ValueRange{args[0]});
          });
      op.getResult(outIdx).replaceUsesWithIf(
          after.outputs()[0], [&](OpOperand &operand) {
            return operand.getOwner() != after.getOperation();
          });
    }
  }

  SmallVector<Type, 4> newOutputTypes;
  for (auto out : newOutputs) {
    newOutputTypes.emplace_back(out.getType());
  }

  // Replace the original generic op
  builder.setInsertionPoint(op);
  auto newGeneric = builder.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/newOutputTypes,
      /*inputs=*/newInputs,
      /*outputs=*/newOutputs,
      /*indexingMaps=*/newIdxMaps,
      /*iteratorTypes=*/
      llvm::to_vector<4>(op.iterator_types().getAsValueRange<StringAttr>()),
      /*doc=*/"",
      /*libraryCall=*/"");
  for (auto attr : op->getAttrs()) {
    if (attr.first != "indexing_maps" && attr.first != "iterator_types" &&
        attr.first != "operand_segment_sizes") {
      newGeneric->setAttr(attr.first, attr.second);
    }
  }
  BlockAndValueMapping bvm;
  op.region().cloneInto(&newGeneric.getRegion(), bvm);

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    op.getResult(i).replaceAllUsesWith(newGeneric.getResult(i));
  }
  op.erase();
}

struct ReorderDimensionsPass
    : public ReorderDimensionsBase<ReorderDimensionsPass> {

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](linalg::GenericOp op) {
      auto candidates = findReorderOperands(checkDim1and3, op);
      reorderDim1and3(op, candidates);
    });
  }
};

std::unique_ptr<mlir::Pass> createReorderDimensionsPass() {
  return std::make_unique<ReorderDimensionsPass>();
}

} // namespace pmlc::dialect::linalgx
