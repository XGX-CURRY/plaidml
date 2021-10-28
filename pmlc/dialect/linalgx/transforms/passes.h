// Copyright 2021 Intel Corporation

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

namespace pmlc::dialect::linalgx {

std::unique_ptr<mlir::Pass> createReorderDimensionsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/linalgx/transforms/passes.h.inc"

} // namespace pmlc::dialect::linalgx
