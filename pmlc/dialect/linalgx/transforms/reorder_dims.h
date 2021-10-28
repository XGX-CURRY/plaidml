// Copyright 2021 Intel Corporation

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/linalgx/ir/ops.h"

namespace pmlc::dialect::linalgx {

// Use the checker to determine which operands should be reordered.
template <typename Checker>
llvm::SmallVector<unsigned, 4> findReorderOperands(const Checker &checker,
                                                   mlir::linalg::GenericOp op) {
  llvm::SmallVector<unsigned, 4> candidates;
  llvm::SmallVector<mlir::AffineMap, 4> idxMaps = llvm::to_vector<4>(
      op.indexing_maps().getAsValueRange<mlir::AffineMapAttr>());
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (checker(idxMaps[i])) {
      candidates.emplace_back(i);
    }
  }
  return candidates;
}

} // namespace pmlc::dialect::linalgx
